"""Track-local scores and the separately selected semantic conflict policy."""
from argparse import Namespace
import numpy as np
import pytest

from pose_pipeline.instance_association import object_consensus
from pose_pipeline.sam3_fusion import GeometricInstances, MapVotes
from pose_pipeline.sam3_guided import recover_instances
from pose_pipeline.semantic_runtime.fusion import apply_semantic_conflict_policy, main, protect_semantic_conflicts
from pose_pipeline.semantic_runtime.multiview import fuse_instances


def evidence(observations, n=60, *, confidence_policy='legacy'):
    ids = np.arange(n)
    masks = np.ones(n, np.int32)
    votes = MapVotes(n, 35)
    streams = [GeometricInstances(), GeometricInstances()]
    frames = []
    high = np.zeros(n, np.float32)
    for fid, (category, quality) in enumerate(observations):
        semantic = np.full(n, category, np.int32)
        confidence = np.full(n, quality, np.float32)
        votes.add(fid, ids, semantic, confidence)
        streams[fid % 2].add(fid, ids, masks, semantic, confidence)
        high = np.maximum(high, confidence)
        frames.append(dict(frame_id=fid, point_ids=ids, mask_ids=masks, semantic=semantic,
                           confidence=confidence, interior=np.ones(n, bool)))
    semantic, confidence, _ = votes.finalize()
    base = {'semantic': semantic, 'confidence': confidence}
    records = [stream.records(confidence_policy, point_scores=high) for stream in streams]
    labels, _, _ = object_consensus(records, [], base, np.zeros((n, 3)))
    return votes, records, base, labels, frames


def test_competing_category_cannot_lend_track_a_high_score():
    # Previously category 1 borrowed .99 from category 2 and filled all 60
    # abstained points with category 1 and confidence .99.
    votes, records, base, labels, _ = evidence([(1, .55), (2, .99), (1, .55)], confidence_policy='track')
    assert not base['semantic'].any()
    assert records[0][0]['mean_point_score'] == pytest.approx(.55)
    assert records[1][0]['mean_point_score'] == pytest.approx(.99)
    assert not labels['semantic'].any()
    assert not labels['confidence'].any()
    default_labels, audit = apply_semantic_conflict_policy(votes, base, labels)
    assert not default_labels['semantic'].any()
    assert audit['policy'] == 'consensus' and audit['enabled'] is False


def test_default_consensus_retains_high_local_confidence_without_running_guard(monkeypatch):
    votes, records, base, before, _ = evidence([(1, .8), (2, .99), (1, .8)], confidence_policy='track')
    assert records[0][0]['mean_point_score'] == pytest.approx(.8)
    assert not base['semantic'].any()
    assert np.all(before['semantic'] == 1)

    def unexpected_guard(*args):
        raise AssertionError('default policy must not execute the experimental guard')
    monkeypatch.setattr('pose_pipeline.semantic_runtime.fusion.protect_semantic_conflicts', unexpected_guard)
    labels, audit = apply_semantic_conflict_policy(votes, base, before)
    assert labels is before
    assert audit == {'policy': 'consensus', 'enabled': False, 'conflicting_abstained_points': 60,
                     'prevented_semantic_fills': 0, 'prevented_instance_assignments': 0}
    assert np.all(labels['confidence'] == pytest.approx(.8))


def test_conflict_guard_blocks_even_a_genuinely_high_score_track_and_stays_unknown():
    votes, _, base, before, frames = evidence([(1, .8), (2, .99), (1, .8)], confidence_policy='track')
    assert not base['semantic'].any()
    assert np.all(before['semantic'] == 1)  # Local scores alone do not protect abstention.
    labels, audit = apply_semantic_conflict_policy(votes, base, before, 'abstain')
    assert audit['policy'] == 'abstain' and audit['enabled'] is True
    assert audit['prevented_semantic_fills'] == 60
    assert audit['prevented_instance_assignments'] == 60
    assert all(not labels[field].any() for field in ('semantic', 'confidence', 'instance'))
    assert np.all(before['semantic'] == 1)  # No caller-owned arrays were mutated.
    current, _ = fuse_instances(60, frames, labels['semantic'],
        {'min_mask_points': 30, 'min_output_points': 50, 'min_point_views': 1,
         'min_group_frames': 2, 'object_score_mode': 'max_point'})
    recovered, _, _ = recover_instances(60, frames, labels['semantic'], labels['instance'], current)
    assert not current.any() and not recovered.any()


def test_conflict_guard_preserves_consistent_evidence_and_existing_labels():
    votes, _, base, before, _ = evidence([(1, .92), (1, .9), (1, .95)], confidence_policy='track')
    labels, audit = protect_semantic_conflicts(votes, base, before)
    assert audit['conflicting_abstained_points'] == 0
    for field in before:
        assert np.array_equal(labels[field], before[field])
    assert np.all(labels['semantic'] == 1)


def test_track_records_require_real_scores_and_reject_bad_confidence():
    tracker = GeometricInstances(min_points=2)
    tracker.add(0, np.arange(2), np.ones(2, int), np.ones(2, int))
    with pytest.raises(ValueError, match='require confidence'):
        tracker.records('track')
    with pytest.raises(ValueError, match='finite confidence'):
        tracker.add(1, np.arange(2), np.ones(2, int), np.ones(2, int), [.8, np.nan])


def test_invalid_conflict_policy_fails_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='semantic_conflict_policy'):
        main(Namespace(arm_root=tmp_path, semantic_conflict_policy='typo'))
    assert not (tmp_path / 'fused').exists()


def test_legacy_reproduces_original_cross_category_score_limitation():
    _, records, base, labels, _ = evidence([(1, .55), (2, .99), (1, .55)], confidence_policy='legacy')
    assert not base['semantic'].any()
    assert records[0][0]['mean_point_score'] == pytest.approx(.99)
    assert np.all(labels['semantic'] == 1)
    assert 'other categories' in records[0][0]['score_scope']


def test_legacy_default_preserves_original_float32_point_mean():
    tracker = GeometricInstances(min_points=2)
    point_ids = np.arange(7)
    own = np.full(7, .5, np.float32)
    high = np.array([.53, .67, .94, .82, .66, .51, .78], np.float32)
    tracker.add(0, point_ids, np.ones(7, int), np.ones(7, int), own)
    records = tracker.records(point_scores=high)
    assert records[0]['mean_point_score'] == float(high[point_ids].mean())
    assert tracker.records('track', point_scores=high)[0]['mean_point_score'] == .5


@pytest.mark.parametrize('scores', [None, [.9, np.nan], [.9, 1.2], [.9]])
def test_legacy_score_policy_keeps_score_validation(scores):
    tracker = GeometricInstances(min_points=2)
    tracker.add(0, np.arange(2), np.ones(2, int), np.ones(2, int), [.8, .9])
    with pytest.raises(ValueError, match='legacy'):
        tracker.records(point_scores=scores)


def test_invalid_confidence_policy_fails_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='semantic_confidence_policy'):
        main(Namespace(arm_root=tmp_path, semantic_confidence_policy='typo'))
    assert not (tmp_path / 'fused').exists()

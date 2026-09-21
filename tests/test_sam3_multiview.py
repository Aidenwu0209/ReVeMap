"""Evidence and ownership contracts for cached multiview instance fusion."""
from copy import deepcopy

import numpy as np
import pytest

from pose_pipeline.sam3_multiview import (
    ConsensusConfig, _cluster, _config, _observations, _view_statistics,
    fuse_instances,
)


def frame(fid, masks, *, confidence=.95):
    masks = np.asarray(masks, np.int32)
    return dict(frame_id=fid, point_ids=np.arange(len(masks), dtype=np.int64),
                mask_ids=masks, semantic=(masks > 0).astype(np.int32),
                confidence=np.full(len(masks), confidence, np.float32),
                interior=np.ones(len(masks), bool))


def cfg(**kwargs):
    return ConsensusConfig(**{'min_mask_points': 1, 'min_output_points': 1, **kwargs})


def fuse(frames, baseline=None, **kwargs):
    n = len(frames[0]['point_ids'])
    baseline = np.ones(n, np.int32) if baseline is None else baseline
    return fuse_instances(n, frames, baseline, cfg(**kwargs))


def test_disjoint_fragments_merge_via_other_complete_views():
    frames = [frame(0, [1]*4+[0]*4), frame(1, [0]*4+[1]*4)]
    frames += [frame(i, [1]*8) for i in (2, 3, 4)]
    result, audit = fuse(frames)
    assert np.all(result > 0)
    assert len(np.unique(result)) == 1
    assert audit['retained_instances'] == 1


def test_repeated_same_class_separation_stays_two_instances():
    result, audit = fuse([frame(i, [1]*4+[2]*4) for i in range(3)])
    assert np.all(result > 0)
    assert len(np.unique(result[:4])) == len(np.unique(result[4:])) == 1
    assert result[0] != result[-1]
    assert audit['retained_instances'] == 2


def test_undersegmented_bridge_is_removed_as_node_and_observer():
    frames = [frame(0, [1]*8)]
    frames += [frame(i, [1]*4+[2]*4) for i in (1, 2, 3)]
    result, audit = fuse(frames)
    assert [(r['frame_id'], r['mask_id']) for r in audit['filtered_masks']] == [(0, 1)]
    assert audit['filtered_masks'][0]['split_frames'] == 3
    assert result[0] != result[-1]
    assert np.all(result > 0)
    assert all(0 not in r['supporting_frames'] for r in audit['objects'])


def test_unknown_mass_cannot_make_a_small_island_contain_whole_object():
    frames = [frame(0, [1]*100)]
    frames += [frame(i, [1]*10+[0]*90) for i in (1, 2, 3)]
    result, audit = fuse(frames)
    assert np.all(result[:10] > 0)
    assert np.all(result[10:] == 0)
    assert not audit['filtered_masks']  # Unknown is also not split evidence.


def test_two_source_frames_alone_cannot_meet_production_support():
    result, audit = fuse([frame(0, [1]*8), frame(1, [1]*8)])
    assert not result.any()
    assert audit['candidate_edges'] == 0


def test_many_masks_in_one_frame_do_not_create_independent_support():
    frames = [frame(0, [1, 1, 2, 2, 3, 3]), frame(1, [1]*6)]
    result, audit = fuse(frames)
    assert not result.any()
    assert audit['candidate_edges'] == 0


def test_component_cannot_link_blocks_transitive_positive_chain():
    positive = np.array([[0, 5, 0], [5, 0, 4], [0, 4, 0]], np.int32)
    negative = np.array([[0, 0, 2], [0, 0, 0], [2, 0, 0]], np.int32)
    groups, audit = _cluster(positive, negative, np.arange(3), np.ones(3, bool), cfg())
    assert groups == [[0, 1], [2]]
    assert audit['accepted_merges'] == 1
    assert audit['component_contradiction_vetoes'] == 1


def test_filtered_node_cannot_be_a_cluster_bridge():
    positive = np.array([[0, 5, 0], [5, 0, 4], [0, 4, 0]], np.int32)
    groups, _ = _cluster(positive, np.zeros((3, 3), np.int32),
                         np.arange(3), np.array([True, False, True]), cfg())
    assert groups == [[0], [2]]


def test_tied_point_ownership_abstains_without_erasing_other_points():
    frames = [frame(i, [1, 1, 1, 0, 0]) for i in (0, 1, 2)]
    frames += [frame(i, [1, 0, 0, 1, 1]) for i in (3, 4, 5)]
    result, audit = fuse(frames)
    assert result[0] == 0
    assert np.all(result[1:] > 0)
    assert result[1] == result[2] != result[3] == result[4]
    assert audit['ambiguous_ownership_points'] == 1


def test_fixed_semantics_split_output_without_mutating_inputs():
    frames = [frame(i, [1]*5) for i in range(3)]
    original_frames = deepcopy(frames)
    semantic = np.array([1, 1, 2, 2, 0], np.int32)
    original_semantic = semantic.copy()
    result, audit = fuse(frames, semantic)
    assert result[0] == result[1] != result[2] == result[3]
    assert result[-1] == 0
    assert not audit['semantic_labels_modified']
    np.testing.assert_array_equal(semantic, original_semantic)
    for current, original in zip(frames, original_frames):
        for key in current:
            np.testing.assert_array_equal(current[key], original[key])


def test_frame_and_point_order_do_not_change_ownership():
    frames = [frame(i, [1]*4+[2]*4) for i in (90, 10, 70)]
    expected, _ = fuse(frames)
    changed = deepcopy(frames[::-1])
    order = np.array([7, 3, 0, 5, 1, 6, 2, 4])
    for f in changed:
        for key in ('point_ids', 'mask_ids', 'semantic', 'confidence', 'interior'):
            f[key] = f[key][order]
    actual, _ = fuse(changed)
    np.testing.assert_array_equal(actual, expected)


def test_low_score_objects_do_not_receive_confident_ownership():
    result, audit = fuse([frame(i, [1]*8, confidence=.79) for i in range(3)])
    assert not result.any()
    assert audit['retained_instances'] == 0


def test_matched_output_keeps_single_view_point_but_rejects_single_view_object():
    # The first object is confirmed in three frames; its fifth point is only
    # measured once. The second object has no corroborating frame at all.
    frames = [frame(0, [1, 1, 1, 1, 1, 2, 2])]
    frames += [frame(i, [1, 1, 1, 1, 0, 0, 0]) for i in (1, 2)]
    strict, strict_audit = fuse(frames)
    matched, matched_audit = fuse(frames, min_point_views=1,
                                  min_group_frames=2, object_score_mode='max_point')
    assert np.all(matched[:5] > 0)
    assert matched[4] == matched[0]
    assert strict[4] == 0
    assert not matched[5:].any()
    assert matched_audit['group_rejections']['insufficient_frames'] == 1
    assert strict_audit['points_below_min_views'] == 1
    assert matched_audit['points_below_min_views'] == 0


def test_object_score_mode_changes_output_gate_but_not_graph():
    frames = [frame(i, [1]*4, confidence=score)
              for i, score in enumerate((.95, .6, .6))]
    mean, mean_audit = fuse(frames, object_score_mode='mean_observations')
    maximum, max_audit = fuse(frames, object_score_mode='max_point')
    assert not mean.any()
    assert np.all(maximum > 0)
    assert mean_audit['group_rejections']['low_score'] == 1
    assert max_audit['group_rejections']['low_score'] == 0
    for key in ('mask_nodes', 'candidate_edges', 'accepted_merges',
                'component_contradiction_vetoes', 'filtered_masks', 'groups_before_output'):
        assert mean_audit[key] == max_audit[key]


def test_point_overlap_statistics_do_not_overflow_16_bits():
    n = 70_000
    frames = [frame(i, np.ones(n, np.int32)) for i in range(3)]
    membership, visibility, _, origins, _ = _observations(n, frames, cfg())
    overlaps, visible, _, filtered, _, _ = _view_statistics(membership, visibility, origins, cfg())
    np.testing.assert_array_equal(overlaps.toarray(), np.full((3, 3), n))
    np.testing.assert_array_equal(visible, np.full((3, 3), n))
    assert not filtered.any()


def test_frame_statistics_do_not_overflow_8_bits():
    result, audit = fuse([frame(i, [1, 1]) for i in range(300)])
    assert np.all(result > 0)
    assert audit['retained_instances'] == 1
    assert len(audit['objects'][0]['supporting_frames']) == 300


@pytest.mark.parametrize('field,value', [
    ('point_ids', [0., 1., 2.]),
    ('point_ids', [0, 0, 2]),
    ('point_ids', [0, 1, 3]),
    ('mask_ids', [1., 1., 1.]),
    ('mask_ids', [-1, 1, 1]),
    ('semantic', [1., 1., 1.]),
    ('semantic', [-1, 1, 1]),
    ('confidence', [np.nan, .9, .9]),
    ('confidence', [-.1, .9, .9]),
    ('confidence', [1.1, .9, .9]),
    ('interior', [np.nan, 1., 1.]),
    ('interior', [2, 1, 1]),
    ('interior', [.2, 1., 1.]),
])
def test_invalid_projected_inputs_are_rejected(field, value):
    bad = frame(0, [1, 1, 1])
    bad[field] = np.asarray(value)
    with pytest.raises(ValueError):
        fuse_instances(3, [bad], np.ones(3, np.int32), cfg())


@pytest.mark.parametrize('fid', [1.5, float('nan'), True])
def test_invalid_frame_id_is_rejected(fid):
    with pytest.raises(ValueError):
        fuse_instances(3, [frame(fid, [1]*3)], np.ones(3, np.int32), cfg())


def test_duplicate_frame_is_rejected():
    with pytest.raises(ValueError, match='duplicate frame'):
        fuse([frame(1, [1]*3), frame(1, [1]*3)])


@pytest.mark.parametrize('n', [3.0, True, -1])
def test_invalid_map_size_is_rejected(n):
    with pytest.raises(ValueError):
        fuse_instances(n, [], np.ones(3, np.int32), cfg())


@pytest.mark.parametrize('config', [
    {'min_point_views': True}, {'min_support_frames': 1.5},
    {'min_mask_points': 0}, {'containment_fraction': .5},
    {'ownership_share': float('nan')}, {'min_output_points': True},
    {'min_group_frames': True}, {'object_score_mode': 'arbitrary'},
])
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        _config(config)


def test_empty_observations_return_unknown_instances():
    result, audit = fuse_instances(3, [], np.ones(3, np.int32), cfg())
    np.testing.assert_array_equal(result, np.zeros(3, np.int32))
    assert audit['retained_instances'] == 0

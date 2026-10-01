"""Measured unknown-point assignment; never overwrite existing labels/owners."""
from collections import Counter
from pathlib import Path
import numpy as np

from ..common import read, write, sha
from ..enhance import export_map, point_ids, validate_labels
from .normalization import canonicalize_name, normalize_name
from .grounding import assess
from ..object_candidates import candidate_anchors, load_candidates


def select_masks(candidates, *, fragments=False):
    """Rank only evidence that satisfies every unchanged acceptance gate.

    A broad or low-score mask must not hide another independently acceptable
    mask. Strict and fragment policies select separately because their gates
    differ; each selected mask still contributes at most one vote per frame.
    """
    rank = lambda x: (x['quality'], x['iou'], x['score'])
    strict = [x for x in candidates if x['score'] >= .5 and x['coverage'] >= .5
              and x['purity'] >= .75 and x['iou'] >= .4 and len(x['own_points']) >= 30]
    fragment = [x for x in candidates if fragments and x['score'] >= .6
                and x['coverage'] >= .8 and x['purity'] >= .5 and x['iou'] >= .5]
    return (max(strict, key=rank, default=None),
            max(fragment, key=rank, default=None),
            max(candidates, key=rank, default=None))


def assign_unknown(base, classes, observations, queries, *, fragments=False, candidate_instance=None):
    """Use original 3D anchors and unique frame evidence, with no extrapolation.

    queries[(frame_id, canonical_name)] contains visible IDs and SAM3 masks.
    All naming views are eligible for independent SAM3 confirmation; heldout
    views never enter this function's support counts.
    """
    n = validate_labels(base)
    anchors = candidate_anchors(base, candidate_instance)
    labels = {k: v.copy() for k, v in base.items()}
    dictionary = dict(classes)
    if dictionary.get('0') != 'unknown' or not set(map(int, np.unique(base['semantic']))).issubset(set(map(int, dictionary))):
        raise ValueError('incomplete class dictionary')
    strength = np.zeros(n, np.float32)
    audits, seen_objects = [], set()
    for ob in observations:
        oid = int(ob['instance_id'])
        if oid <= 0 or oid in seen_objects:
            raise ValueError('duplicate or invalid instance')
        seen_objects.add(oid)
        vote = ob['votes']['quality_canonical']
        frames = vote['frames']
        if len(frames) != len(set(frames)):
            raise ValueError('duplicate naming frame')
        names = vote['labels']
        if len(names) != len(frames):
            raise ValueError('frame/name count mismatch')
        counts = Counter(name for name in names if name != 'unknown').most_common()
        proposed = counts[0][0] if counts and counts[0][1] >= 2 and (len(counts) == 1 or counts[0][1] > counts[1][1]) else 'unknown'
        if proposed != vote['name']:
            raise ValueError('consensus does not match naming votes')
        name = canonicalize_name(proposed)
        owned = anchors['instance'] == oid
        unknown = owned & (base['semantic'] == 0)
        known_categories = np.unique(base['semantic'][(base['instance'] == oid) & (base['semantic'] > 0)])
        if (not unknown.any() or name == 'unknown' or normalize_name(proposed)['is_part']
                or any(canonicalize_name(dictionary[str(sid)]) != name for sid in known_categories)):
            continue
        strict_support = np.zeros(n, np.uint16)
        fragment_support = np.zeros(n, np.uint16)
        strict_frames, fragment_frames, details = [], [], []
        for fid in frames:
            query = queries.get((fid, name))
            if query is None:
                continue
            visible = point_ids(query['visible'], n)
            candidates = []
            for mask in query['candidates']:
                pts = point_ids(mask['points'], n)
                score = float(mask['score'])
                if not np.isfinite(score) or not 0 <= score <= 1 or not np.isin(pts, visible).all():
                    raise ValueError('invalid or invisible SAM3 evidence')
                candidates.append(assess(pts, visible, anchors['instance'], oid, score))
            strict_mask, fragment_mask, raw_best = select_masks(candidates, fragments=fragments)
            if strict_mask is not None:
                strict_support[strict_mask['own_points']] += 1
                strict_frames.append(fid)
            if fragment_mask is not None:
                fragment_support[fragment_mask['own_points']] += 1
                fragment_frames.append(fid)
            summarize = lambda mask: {k: v for k, v in mask.items()
                                     if k not in ('own_points', 'mask_points')} if mask is not None else None
            best = strict_mask if strict_mask is not None else (fragment_mask if fragment_mask is not None else raw_best)
            details.append({'frame_id': fid, 'strict': strict_mask is not None,
                            'fragment': fragment_mask is not None, 'selection': 'eligible_then_quality',
                            'evidence': summarize(best), 'strict_evidence': summarize(strict_mask),
                            'fragment_evidence': summarize(fragment_mask)})
        strict_points = unknown & (strict_support >= 2)
        if len(strict_frames) < 2 or strict_points.sum() < 50:
            strict_points[:] = False
        # Match the validated sequential policy: fragment fill only remaining points.
        extra = unknown & ~strict_points & (fragment_support >= 3)
        if len(fragment_frames) < 3 or extra.sum() < 50:
            extra[:] = False
        eligible = strict_points | extra
        if eligible.any():
            lookup = {canonicalize_name(v): int(k) for k, v in dictionary.items()}
            if name not in lookup:
                lookup[name] = max(map(int, dictionary)) + 1
                dictionary[str(lookup[name])] = name
            labels['semantic'][eligible] = lookup[name]
            labels['instance'][eligible] = anchors['instance'][eligible]
            strength[strict_points] = strict_support[strict_points] / len(frames)
            strength[extra] = fragment_support[extra] / 3
        audits.append({'instance_id': oid, 'name': name, 'strict_points': int(strict_points.sum()),
                       'fragment_points': int(extra.sum()), 'views': details})
    known = base['semantic'] > 0
    assert np.array_equal(labels['semantic'][known], base['semantic'][known])
    established = base['instance'] > 0
    assert np.array_equal(labels['instance'][established], base['instance'][established])
    assert not np.any((labels['instance'] != base['instance']) & (labels['semantic'] == 0))
    assert np.array_equal(labels['confidence'], base['confidence'])
    return labels, dictionary, strength, audits


def apply(workspace, *, fragments=False):
    root = Path(workspace)
    out = root / ('refined-fragments' if fragments else 'refined')
    out.mkdir(exist_ok=False)
    records = read(root / 'grounding/RECORDS.json')
    decisions = read(root / 'DECISIONS.json')
    results = []
    for scene in read(root / 'INPUT_PLAN.json')['scenes']:
        inp = root / 'inputs' / scene
        with np.load(inp / 'base.npz', allow_pickle=False) as z:
            base = {k: v.copy() for k, v in z.items()}
        xyz = np.load(inp / 'target.npz', allow_pickle=False)['xyz']
        if xyz.shape != (validate_labels(base), 3) or not np.isfinite(xyz).all():
            raise ValueError('invalid fixed target')
        queries = {}
        for row in records:
            if row['scene'] != scene:
                continue
            key = (row['frame_id'], row['name'])
            if key in queries:
                raise ValueError('duplicate grounding query')
            p = (root / row['file']).resolve()
            if not p.is_relative_to(root.resolve()) or sha(p) != row['sha256']:
                raise ValueError('grounding evidence changed or escaped workspace')
            with np.load(p, allow_pickle=False) as z:
                queries[key] = {'visible': z['visible'].copy(), 'candidates': [
                    {'points': z[f'points_{i}'].copy(), 'score': float(s)} for i, s in enumerate(z['scores'])]}
        labels, classes, strength, audit = assign_unknown(base, read(inp / 'classes.json'),
            [ob for ob in decisions if ob['scene'] == scene], queries, fragments=fragments,
            candidate_instance=load_candidates(inp))
        dest = out / scene
        dest.mkdir(parents=True)
        np.savez_compressed(dest / 'map_labels.npz', **labels)
        write(dest / 'classes.json', classes)
        export_map(xyz, labels, classes, dest, strength)
        result = {'status': 'completed', 'scene': scene, 'changed_points': int(np.sum(labels['semantic'] != base['semantic'])),
                  'known_labels_preserved': True, 'geometry_modified': False,
                  'instance_ids_preserved': bool(np.array_equal(labels['instance'], base['instance'])),
                  'existing_instance_ids_preserved': True,
                  'promoted_candidate_points': int(np.sum(labels['instance'] != base['instance'])),
                  'GT_used': False, 'fragment_policy': fragments,
                  'scope': 'fixed geometry semantic refinement; no SLAM or SGA inference', 'objects': audit}
        write(dest / 'RESULT.json', result)
        from ...artifacts import write_artifact_manifest
        write_artifact_manifest(dest, map_path=dest / 'semantic_labeled.ply',
            classes_path=dest / 'classes.json', result_path=dest / 'RESULT.json',
            manifest_path=inp / 'manifest.json', trajectory_path=inp / 'trajectory.json',
            extra_files={'labels': dest / 'map_labels.npz'})
        results.append(result)
    write(out / 'RESULTS.json', results)
    write(out / 'PREDICTIONS_LOCK.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    return results

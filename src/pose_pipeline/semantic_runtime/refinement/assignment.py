"""Measured unknown-point assignment; never overwrite existing labels/owners."""
from collections import Counter
from pathlib import Path
import numpy as np

from ..common import read, write, sha
from ..enhance import export_map, point_ids, validate_labels
from .normalization import canonicalize_name, normalize_name
from .grounding import assess


def assign_unknown(base, classes, observations, queries, *, fragments=False):
    """Use original 3D anchors and unique frame evidence, with no extrapolation.

    queries[(frame_id, canonical_name)] contains visible IDs and SAM3 masks.
    All naming views are eligible for independent SAM3 confirmation; heldout
    views never enter this function's support counts.
    """
    n = validate_labels(base)
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
        owned = base['instance'] == oid
        if not owned.any() or ob['semantic_id'] != 0 or name == 'unknown' or normalize_name(proposed)['is_part']:
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
                candidates.append(assess(pts, visible, base['instance'], oid, score))
            best = max(candidates, key=lambda x: (x['quality'], x['iou'], x['score'])) if candidates else None
            strict = bool(best and best['score'] >= .5 and best['coverage'] >= .5 and best['purity'] >= .75 and best['iou'] >= .4 and len(best['own_points']) >= 30)
            fragment = bool(fragments and best and best['score'] >= .6 and best['coverage'] >= .8 and best['purity'] >= .5 and best['iou'] >= .5)
            if strict:
                strict_support[best['own_points']] += 1
                strict_frames.append(fid)
            if fragment:
                fragment_support[best['own_points']] += 1
                fragment_frames.append(fid)
            details.append({'frame_id': fid, 'strict': strict, 'fragment': fragment,
                            'evidence': {k: v for k, v in best.items() if k not in ('own_points', 'mask_points')} if best else None})
        unknown = owned & (base['semantic'] == 0)
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
            strength[strict_points] = strict_support[strict_points] / len(frames)
            strength[extra] = fragment_support[extra] / 3
        audits.append({'instance_id': oid, 'name': name, 'strict_points': int(strict_points.sum()),
                       'fragment_points': int(extra.sum()), 'views': details})
    known = base['semantic'] > 0
    assert np.array_equal(labels['semantic'][known], base['semantic'][known])
    assert np.array_equal(labels['instance'], base['instance'])
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
            [ob for ob in decisions if ob['scene'] == scene], queries, fragments=fragments)
        dest = out / scene
        dest.mkdir(parents=True)
        np.savez_compressed(dest / 'map_labels.npz', **labels)
        write(dest / 'classes.json', classes)
        export_map(xyz, labels, classes, dest, strength)
        result = {'scene': scene, 'changed_points': int(np.sum(labels['semantic'] != base['semantic'])),
                  'known_labels_preserved': True, 'geometry_modified': False,
                  'instance_ids_preserved': True, 'GT_used': False, 'fragment_policy': fragments,
                  'scope': 'fixed geometry semantic refinement; no SLAM or SGA inference', 'objects': audit}
        write(dest / 'RESULT.json', result)
        results.append(result)
    write(out / 'RESULTS.json', results)
    write(out / 'PREDICTIONS_LOCK.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    return results

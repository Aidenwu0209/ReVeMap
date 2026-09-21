"""Convert archived, real SAM3/VLM outputs into a portable P2 replay bundle.

This adapter reads no GT. It extracts all mask alternatives, not a preselected
winning result. The replay entry recomputes T1, P2 and naming decisions.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from pose_pipeline.semantic_runtime.common import read, sha, write


def build(root, scene, model_run, output, cache_overlay=None):
    output.mkdir(parents=True, exist_ok=False)
    files, sources = {}, {}
    def read_json(path):
        sources[str(path)] = sha(path)
        return read(path)
    def load(path):
        if not path.exists() and cache_overlay is not None:
            path = cache_overlay / path.relative_to(root)
        sources[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}
    def save(name, **data):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        files[name] = sha(path)
        return name
    def extract(path, projection, prefix='points'):
        z = load(path)
        height, width = map(int, z['depth_shape'])
        if (height, width) != tuple(projection['depth_shape']):
            raise ValueError('query and projection resolution differs')
        data = {prefix+'_scores': z['scores']}
        for mi, packed in enumerate(z['packed_masks']):
            mask = np.unpackbits(packed, count=height*width).reshape(height, width).astype(bool)
            safe = np.zeros_like(mask)
            safe[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                               & mask[1:-1, :-2] & mask[1:-1, 2:])
            data[f'{prefix}_{mi}'] = np.unique(projection['point_ids'][safe[projection['row'], projection['col']]])
        return data
    birth = root / 'semantic_birth_20260914_v1' / scene
    surface = root / 'semantic_surface_support_20260914_v1' / scene
    target = load(root / 'preprocessing_validation_20260911_v1/inputs' / scene / 'target.npz')
    original = load(birth / 'A0_baseline/map_labels.npz')
    plan = read_json(birth / 'PROMPT_PLAN.json')
    bundle = {'schema_version': 1, 'scene': scene,
              'base': save('base.npz', **original), 'original': 'base.npz',
              'target': save('target.npz', xyz=target['xyz']),
              'geometry_xyz_sha256': hashlib.sha256(np.ascontiguousarray(target['xyz']).tobytes()).hexdigest(),
              'classes': read_json(birth / 'linked/anchored_instance_point/classes.json'),
              'seed_regions': {str(i): row['point_ids'] for i, row in enumerate(plan['directed'])},
              'identity_queries': [], 'pair_views': [], 'models': {}, 'files': files}
    for i, row in enumerate(read_json(birth / 'linked/QUERIES.json')):
        projection = load(birth / 'linked/images' / f'{row["frame_id"]:06}_projection.npz')
        masks = extract(birth / 'linked/queries' / (row['name'] + '.npz'), projection)
        masks['scores'] = masks.pop('points_scores')
        evidence = save(f'identity/{i:04}.npz', visible=projection['point_ids'], **masks)
        bundle['identity_queries'].append({k: row[k] for k in ['name', 'region', 'view_index', 'frame_id', 'chosen_mask_index']})
        bundle['identity_queries'][-1]['evidence'] = evidence
    pair_plan = read_json(surface / 'PAIR_PLAN.json')
    for pair in pair_plan['ready_pairs']:
        for view in pair['views']:
            fid, a, b = view['frame_id'], pair['a'], pair['b']
            projection = load(surface / 'pair_queries/images' / f'{fid:06}_projection.npz')
            data = {'visible': projection['point_ids']}
            for side, oid in [('left', a), ('right', b)]:
                data.update(extract(surface / 'pair_queries/queries' / f'pair_{a}_{b}_{fid:06}_from_{oid}.npz', projection, side))
            bundle['pair_views'].append({'pair': [a, b], 'frame_id': fid,
                'evidence': save(f'pairs/{a}_{b}_{fid:06}.npz', **data)})
    if model_run:
        # The latest Mage/Joy benchmark includes a fresh Qwen control; all its
        # models share exactly the same fixed map and projection interface.
        ground = model_run / 'ground_inputs' / scene
        cv = {(x['instance_id'], x['frame_id']): x for x in read_json(ground / 'crossview_selection.json')}
        for arm in read_json(model_run / 'ARMS.json'):
            mid = arm['id']
            records = read_json(model_run / 'runs' / mid / 'primary/RECORDS.json')
            names = [{k: row[k] for k in ['instance_id', 'frame_id', 'role', 'crop_mode', 'raw_response']}
                     for row in records if row['scene'] == scene and row['pool'] == 'legacy_unknown_objects']
            bundle['models'][mid] = {'naming': names, 'grounding': []}
        records = read_json(model_run / 'GROUNDING_RECORDS.json')
        for i, row in enumerate(records):
            if row['scene'] != scene:
                continue
            oid, fid = row['instance_id'], row['frame_id']
            projection = load(ground / 'projections' / f'{fid:06}.npz')
            ref = cv[(oid, fid)]
            reference = np.empty(0, np.int64)
            if ref['passed']:
                selected = ref['selected']
                candidates = extract(ground / 'queries' / (selected['query'] + '.npz'), projection)
                reference = candidates[f'points_{selected["mask_index"]}']
            masks = extract(model_run / row['file'], projection)
            masks['scores'] = masks.pop('points_scores')
            evidence = save(f'grounding/{i:04}.npz', visible=projection['point_ids'], reference=reference, **masks)
            for mid in row['models']:
                bundle['models'][mid]['grounding'].append({k: row[k] for k in ['instance_id', 'frame_id', 'label', 'role']})
                bundle['models'][mid]['grounding'][-1]['evidence'] = evidence
    write(output / 'BUNDLE.json', bundle)
    # Conversion-time hashes are post-hoc provenance, not a new preregistration.
    write(output / 'SOURCE_HASHES.json', {'stage': 'post-hoc conversion', 'files': sources, 'GT_read': False})
    return output / 'BUNDLE.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparisons-root', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--model-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-overlay', type=Path, help='separate mirror for missing archived projection files')
    args = parser.parse_args()
    print(build(args.comparisons_root.resolve(), args.scene, args.model_run.resolve() if args.model_run else None,
                args.output.resolve(), args.cache_overlay.resolve() if args.cache_overlay else None))

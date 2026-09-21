"""Export cached-SAM3 fusion labels without changing source vertex properties."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
from plyfile import PlyData, PlyElement

R = Path(__file__).resolve().parent
PREVIOUS = R.parent / 'sam3_sga_20260912_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='consensus_instances')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--allow-semantic-change', action='store_true',
                        help='Explicitly permit a future semantic-changing experiment.')
    parser.add_argument('--allow-unknown-instances', action='store_true',
                        help='Explicit experimental contract: independent instance and semantic IDs.')
    args = parser.parse_args()
    run = R / args.stage / args.scene
    reference = PREVIOUS / 'objects_geometry' / args.scene
    if args.scene.startswith('orbbec/'):
        baseline = R.parent / 'sgf_sga_orbbec_4812_20260910_v1/baseline/refused.ply'
    else:
        records = json.loads((PREVIOUS / 'inputs/TARGET_PROVENANCE.json').read_text())
        baseline = Path(next(r for r in records if r['key'] == args.scene)['baseline'])
    labels, result, output = run / 'map_labels.npz', run / 'result.json', run / 'map'
    report = json.loads(result.read_text())
    source = PlyData.read(baseline)
    vertex = source['vertex'].data
    xyz = np.stack([vertex[k] for k in ('x', 'y', 'z')], axis=1)
    xyz_hash = hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()
    if report['status'] != 'completed' or report.get('geometry_xyz_sha256') != xyz_hash:
        raise ValueError('Completed labels must match exact original XYZ precision and point order.')
    with np.load(labels) as data:
        semantic, instance, confidence = (data[k].copy() for k in ('semantic', 'instance', 'confidence'))
    if any(a.shape != (len(vertex),) for a in (semantic, instance, confidence)):
        raise ValueError('Label count differs from original vertex count.')
    if semantic.dtype.kind not in 'iu' or instance.dtype.kind not in 'iu':
        raise ValueError('Category and instance IDs must be integer arrays.')
    if np.any(semantic < 0) or np.any(instance < 0) or not np.isfinite(confidence).all():
        raise ValueError('Invalid labels or confidence.')
    if np.any((instance > 0) & (semantic == 0)) and not args.allow_unknown_instances:
        raise ValueError('Instance cannot have unknown category.')
    with np.load(reference / 'map_labels.npz') as previous:
        same_semantic = np.array_equal(semantic, previous['semantic'])
        same_confidence = np.array_equal(confidence, previous['confidence'])
    if not args.allow_semantic_change and not same_semantic:
        raise ValueError('Instance-only experiment changed semantic labels.')
    fields = [('semantic_id', '<i4'), ('instance_id', '<i4'), ('semantic_confidence', '<f4')]
    if any(k in vertex.dtype.names for k, _ in fields):
        raise ValueError('Use original geometry, which must not already contain label fields.')
    source_sha = sha(baseline)
    output.mkdir(parents=True, exist_ok=False)
    labeled = np.empty(len(vertex), dtype=vertex.dtype.descr + fields)
    for key in vertex.dtype.names:
        labeled[key] = vertex[key]
    labeled['semantic_id'], labeled['instance_id'], labeled['semantic_confidence'] = semantic, instance, confidence

    def write(path, vertices):
        elements = [PlyElement.describe(vertices, 'vertex') if e.name == 'vertex' else e
                    for e in source.elements]
        PlyData(elements, text=False).write(str(path))

    write(output / 'map_labeled.ply', labeled)
    check = PlyData.read(output / 'map_labeled.ply')['vertex'].data
    preserved = {k: bool(np.array_equal(check[k], vertex[k], equal_nan=True))
                 for k in vertex.dtype.names}
    if not all(preserved.values()) or sha(baseline) != source_sha:
        raise RuntimeError('Original vertex property preservation failed.')
    for key in ('semantic_id', 'instance_id'):
        colored = labeled.copy()
        ids = labeled[key].astype(np.int64)
        for channel, multiplier in zip(('red', 'green', 'blue'), (73, 151, 199)):
            colored[channel] = np.where(ids > 0, 50 + ids * multiplier % 206, 90).astype(np.uint8)
        write(output / f'map_{key}.ply', colored)
    np.save(output / 'semantic.npy', semantic)
    np.save(output / 'instance.npy', instance)
    copied = {}
    for filename in ['classes.json', 'objects.json', 'scene_graph.json']:
        origin = run / filename
        if filename == 'classes.json' and not origin.exists():
            origin = reference / filename
        if origin.exists():
            shutil.copy2(origin, output / filename)
            copied[filename] = {'path': str(origin), 'sha256': sha(origin)}
    receipt = {
        'baseline': str(baseline), 'baseline_sha256': source_sha,
        'geometry_xyz_sha256': xyz_hash, 'original_vertex_properties_preserved': preserved,
        'point_count': len(vertex), 'semantic_coverage': float(np.mean(semantic > 0)),
        'instance_coverage': float(np.mean(instance > 0)), 'inference_result': str(result),
        'unknown_semantic_instance_points': int(np.sum((instance > 0) & (semantic == 0))),
        'unknown_instances_explicitly_allowed': args.allow_unknown_instances,
        'result_sha256': sha(result), 'label_sha256': sha(labels), 'copied_metadata': copied,
        'reference': str(reference), 'semantic_identical_to_objects_geometry': same_semantic,
        'confidence_identical_to_objects_geometry': same_confidence,
        'sam3_inference_this_run': report.get('new_model_inference', False),
        'sga_inference_executed_this_run': report.get('sga_inference_executed', False),
        'complete_full_sequence': report.get('complete_full_sequence', False),
        'quality_accepted': False,
        'colored_ply_note': 'map_labeled preserves RGB; semantic/instance display PLY files recolor only RGB.'}
    (output / 'MAP_CONTRACT.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

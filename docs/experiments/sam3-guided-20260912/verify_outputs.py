"""Independent local artifact checks for the three final CPU experimental arms."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from plyfile import PlyData

R = Path(__file__).resolve().parent
STAGES = ('guided_recovery_maskveto', 'unknown_contract', 'raw_unknown')


def read(p):
    return json.loads(p.read_text())


def main():
    checks = []
    for job in read(R / 'inputs/JOBS.json'):
        key = job['key']
        croot = R.parent / 'sam3_sga_20260912_v1/objects_geometry' / key
        r3root = R.parent / 'sam3_multiview_20260912_v1/consensus_matched' / key
        with np.load(croot / 'map_labels.npz', allow_pickle=False) as z:
            baseline = {k: z[k].copy() for k in z.files}
        with np.load(r3root / 'map_labels.npz', allow_pickle=False) as z:
            previous = z['instance'].copy()
        for stage in STAGES:
            root = R / stage / key
            result, contract = read(root / 'result.json'), read(root / 'map/MAP_CONTRACT.json')
            assert result['status'] == 'completed'
            assert result['selected_frame_ids'] == job['selected_frame_ids']
            assert result['processed_frames'] == len(job['selected_frame_ids'])
            with np.load(root / 'map_labels.npz', allow_pickle=False) as z:
                labels = {k: z[k].copy() for k in z.files}
            unchanged = {k: bool(np.array_equal(v, labels[k])) for k, v in baseline.items() if k != 'instance'}
            assert all(unchanged.values())
            source = PlyData.read(contract['baseline'])['vertex'].data
            output = PlyData.read(root / 'map/map_labeled.ply')['vertex'].data
            assert len(output) == job['expected_points']
            preserved = {k: bool(np.array_equal(source[k], output[k], equal_nan=True)) for k in source.dtype.names}
            assert all(preserved.values())
            xyz = np.stack([output[k] for k in ('x', 'y', 'z')], axis=1)
            assert np.isfinite(xyz).all()
            assert hashlib.sha256(xyz.tobytes()).hexdigest() == job['geometry_xyz_sha256']
            for label, field in [('instance', 'instance_id'), ('semantic', 'semantic_id'), ('confidence', 'semantic_confidence')]:
                assert np.array_equal(labels[label], output[field])
            objects = read(root / 'objects.json')
            assert len({o['instance_id'] for o in objects}) == len(objects)
            assert set(o['instance_id'] for o in objects) == set(np.unique(labels['instance'])) - {0}
            for obj in objects:
                points = labels['instance'] == obj['instance_id']
                assert int(points.sum()) == obj['point_count']
                values, counts = np.unique(labels['semantic'][points], return_counts=True)
                actual = {str(int(k)): int(v) for k, v in zip(values, counts)}
                assert actual == obj['semantic_histogram']
            graph = read(root / 'scene_graph.json')
            assert graph['relations'] == [] and graph['new_relation_prediction_executed'] is False
            views = read(root / 'MAP_VIEW_RECEIPT.json')
            assert views['identical_original_geometry'] and views['full_bounds_used'] and views['no_roi_cropping']
            images = {}
            for name in ('map_instance_comparison.png', 'map_semantic_comparison.png'):
                with Image.open(root / name) as image:
                    image.verify()
                with Image.open(root / name) as image:
                    assert image.width > 1500 and image.height > 800
                    images[name] = list(image.size)
            recovery = None
            if stage == 'guided_recovery_maskveto':
                assert np.array_equal(previous[previous > 0], labels['instance'][previous > 0])
                recovered = np.flatnonzero((previous == 0) & (labels['instance'] > 0))
                with np.load(root / 'perpoint_provenance.npz', allow_pickle=False) as z:
                    assert np.array_equal(recovered, z['point_id'])
                    assert len(np.unique(z['point_id'])) == len(recovered)
                    assert np.array_equal(labels['instance'][recovered], z['destination_instance_id'])
                    assert np.array_equal(baseline['instance'][recovered], z['guide_instance_id'])
                    pairs = np.unique(np.column_stack([z['support_point_id'], z['support_frame_id']]), axis=0)
                    ids, counts = np.unique(pairs[:, 0], return_counts=True)
                    assert np.array_equal(ids, recovered) and np.all(counts >= 2)
                recovery = {'previous_owners_unchanged': True, 'points': len(recovered), 'distinct_frames_at_least_two': True}
            checks.append({'stage': stage, 'key': key, 'points': len(output),
                           'non_instance_arrays_preserved': unchanged, 'original_vertex_fields_preserved': preserved,
                           'object_inventory_exact': True, 'graph_relations_not_claimed': True,
                           'figures': images, 'recovery': recovery,
                           'unknown_semantic_instance_points': int(np.sum((labels['instance'] > 0) & (labels['semantic'] == 0)))})
    for evaluator in ('evaluate_0030.py', 'evaluate_instances_0030.py'):
        assert (R / evaluator).read_bytes() == (R.parent / 'sam3_multiview_20260912_v1' / evaluator).read_bytes()
    receipt = {'all_passed': True, 'checked_maps': len(checks), 'expected_maps': 15,
               'checks': checks, 'legacy_evaluators_unchanged': True,
               'scope': 'artifact validity and preservation, not semantic or instance accuracy'}
    (R / 'FINAL_OUTPUT_AUDIT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'all_passed': True, 'checked_maps': len(checks)}))


if __name__ == '__main__':
    main()

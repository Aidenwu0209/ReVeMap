"""Run serialized, same-input SAM3 and frozen-geometry fusion comparisons."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fusion_worker(reference, semantic, output):
    """Keep the unchanged fusion, but seal the actual partial replay scope."""
    output.mkdir(parents=True, exist_ok=False)
    (output / 'mapping').mkdir()
    write(output / 'mapping/mapping_result.json', read(reference / 'mapping/mapping_result.json'))
    (output / 'semantic').symlink_to(semantic.resolve(), target_is_directory=True)
    from pose_pipeline import sam3_export
    original = sam3_export.export

    def scoped_export(cloud, labels, result, destination):
        report = read(result)
        report.update(complete_full_sequence=False, raw_window_complete=False,
                      pipeline_scope='fresh selected SAM3 evidence on fixed saved geometry; no fresh SLAM/VLM/refinement')
        write(result, report)
        return original(cloud, labels, result, destination)

    sam3_export.export = scoped_export
    try:
        from pose_pipeline.semantic_runtime.fusion import main
        main(argparse.Namespace(arm_root=output))
    finally:
        sam3_export.export = original


def suite(plan_path, output):
    from pose_pipeline.contracts import load_manifest, write_manifest
    import numpy as np
    from compare_sam3_outputs import compare
    plan = read(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'PLAN.json', plan)
    results = []
    for index, scene in enumerate(plan['scenes']):
        dest = output / scene['name']
        dest.mkdir()
        reference = Path(scene['reference'])
        geometry = read(reference / 'mapping/mapping_result.json')
        manifest = load_manifest(Path(geometry['manifest']))
        if scene.get('window'):
            start, length = scene['window']
            manifest = replace(manifest, frames=manifest.frames[start:start + length])
        manifest_path = dest / 'manifest.json'
        write_manifest(manifest_path, manifest)
        protected = [Path(geometry[k]) for k in ('manifest', 'trajectory', 'final_cloud')]
        protected += [p for f in manifest.frames for p in (f.color_path, f.depth_path)]
        inputs = {str(p): sha(p) for p in protected}
        write(dest / 'INPUTS.json', inputs)
        order = ['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline']
        for arm in order:
            env = {**os.environ, 'PYTHONPATH': plan[arm + '_source'],
                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
            with (dest / (arm + '.log')).open('x') as log:
                subprocess.run([plan['sam3_python'], str(Path(__file__).with_name('profile_sam3.py')),
                                '--manifest', str(manifest_path), '--runtime', plan['runtime'],
                                '--output', str(dest / arm), '--stride', '5'], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            write(output / 'PROGRESS.json', {'scene': scene['name'], 'completed_arm': arm,
                                           'scenes_finished': len(results)})
            print(scene['name'], arm, 'completed', flush=True)
        parity = compare(dest / 'baseline', dest / 'candidate')
        write(dest / 'SAM3_PARITY.json', parity)
        if not parity['passed']:
            raise RuntimeError('SAM3 evidence mismatch: ' + scene['name'])
        for arm in ('baseline', 'candidate'):
            env = {**os.environ, 'PYTHONPATH': plan[arm + '_source'],
                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
            with (dest / (arm + '-fusion.log')).open('x') as log:
                subprocess.run([plan['cpu_python'], __file__, '--fusion', str(reference),
                                '--semantic', str(dest / arm), '--output', str(dest / (arm + '-fusion'))],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(dest / 'baseline-fusion/fused/map_labels.npz') as a, np.load(dest / 'candidate-fusion/fused/map_labels.npz') as b:
            equal = {k: a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a.files}
        maps = {arm: sha(dest / (arm + '-fusion') / 'fused/export/map_labeled.ply') for arm in ('baseline', 'candidate')}
        unchanged = all(sha(p) == value for p, value in inputs.items())
        result = {'scene': scene['name'], 'window': scene.get('window'), 'raw_frames': len(manifest.frames),
                  'first_frame': manifest.frames[0].frame_id, 'last_frame': manifest.frames[-1].frame_id,
                  'order': order, 'sam3_parity': parity, 'labels_equal': equal, 'map_sha256': maps,
                  'inputs_unchanged': unchanged, 'baseline': read(dest / 'baseline/PROFILE_SUMMARY.json'),
                  'candidate': read(dest / 'candidate/PROFILE_SUMMARY.json'),
                  'scope': 'SAM3 and fusion only; fixed saved geometry; no new GT accuracy claim'}
        result['passed'] = all(equal.values()) and len(set(maps.values())) == 1 and unchanged
        write(dest / 'RESULT.json', result)
        results.append(result)
        write(output / 'RESULTS.json', results)
        print(scene['name'], 'PASSED' if result['passed'] else 'FAILED', flush=True)
        if not result['passed']:
            raise RuntimeError('fusion or input parity failure')
    write(output / 'COMPLETE.json', {'scenes': len(results), 'passed': all(r['passed'] for r in results)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path)
    p.add_argument('--fusion', type=Path)
    p.add_argument('--semantic', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.fusion:
        fusion_worker(a.fusion, a.semantic, a.output)
    else:
        suite(a.plan, a.output)

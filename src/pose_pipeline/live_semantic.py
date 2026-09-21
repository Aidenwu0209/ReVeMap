"""GUI bridge to the public developnew runtime; never substitutes cached outputs."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .live_io import atomic_json, read_json


def prepare_refinement(output, manifest, runtime):
    """Bridge this run's exact point ordering into the validated refinement contract."""
    root = output / 'refinement'
    inp = root / 'inputs' / 'capture'
    inp.mkdir(parents=True, exist_ok=False)
    geom = read_json(output / 'mapping/mapping_result.json')
    for source, name in [(manifest, 'manifest.json'), (Path(geom['trajectory']), 'trajectory.json'),
                         (output / 'fused/target.npz', 'target.npz'),
                         (output / 'fused/map_labels.npz', 'base.npz'),
                         (output / 'fused/classes.json', 'classes.json')]:
        shutil.copyfile(source, inp / name)
    shutil.copyfile(runtime, root / 'runtime.json')
    atomic_json(inp / 'INPUT.json', {'rgb_registration': 'already_registered',
                                    'source': str(output), 'GT_used': False})
    atomic_json(root / 'INPUT_PLAN.json', {'scenes': ['capture']})
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--schedule', choices=['serial', 'parallel'], default='serial')
    p.add_argument('--vlm', default='qwen3vl_2b_nf4')
    p.add_argument('--refine', action='store_true')
    args = p.parse_args()
    args.stride = 5
    def terminate(*_):
        raise KeyboardInterrupt('GUI cancelled')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    from .semantic_runtime.pipeline import run
    result = run(args)
    output = args.output.resolve()
    geom = read_json(output / 'mapping/mapping_result.json')
    final = result['map']
    classes = output / "fused/classes.json"
    if args.refine:
        root = prepare_refinement(output, args.manifest, args.runtime)
        cfg = read_json(args.runtime)
        # Same process group as bridge: GUI cancellation reaches active refinement.
        for stage, python in [('prepare', cfg['cpu_python']), ('name', cfg['vlm_python']),
                              ('decide', cfg['cpu_python']), ('ground', cfg['sam3_python']),
                              ('apply', cfg['cpu_python'])]:
            atomic_json(output / 'GUI_STAGE.json', {'stage': 'refine/' + stage})
            child = subprocess.Popen([python, '-m', 'pose_pipeline.semantic_runtime.refinement',
                                      '--workspace', str(root), '--stage', stage])
            try:
                code = child.wait()
                if code:
                    raise RuntimeError(f'refinement/{stage} failed ({code})')
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
        final = str(root / 'refined/capture/semantic_labeled.ply')
        classes = root / 'refined/capture/classes.json'
    if not Path(final).is_file():
        raise RuntimeError('Missing final labeled PLY')
    atomic_json(output / 'GUI_STAGE.json', {'stage': 'completed'})
    atomic_json(output / 'GUI_RESULT.json', {'final_cloud': final, 'trajectory': geom['trajectory'],
                'classes': str(classes), 'raw_map': result['map'], 'names': result['names'], 'refinement': args.refine,
                'scope': 'run-sam3 plus optional direct unknown-point refinement; no offline P2'})


if __name__ == '__main__':
    main()

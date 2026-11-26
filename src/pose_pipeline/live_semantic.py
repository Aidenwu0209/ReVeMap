"""GUI bridge to the public developnew runtime; never substitutes cached outputs."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .live_io import atomic_json, read_json, guard_gui_parent


def prepare_refinement(output, manifest, runtime, model="qwen3vl_2b_nf4"):
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
    from .semantic_runtime.common import LIVE_NAMERS
    cfg = read_json(runtime)
    if model in LIVE_NAMERS:
        cfg['refinement_namer'] = model
    atomic_json(root / 'runtime.json', cfg)
    candidate_path = output / 'fused/candidates.npz'
    if candidate_path.exists():
        from .contracts import sha256_file
        receipt = read_json(output / 'fused/CANDIDATES.json')
        if (sha256_file(candidate_path) != receipt['candidates_sha256']
                or sha256_file(inp / 'base.npz') != receipt['labels_sha256']):
            raise ValueError('Candidate evidence does not match refinement base')
        shutil.copyfile(candidate_path, inp / 'candidates.npz')
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
    p.add_argument("--map-completion", choices=("anchored", "off"), default="anchored",
                      help="extend existing instances onto multi-view depth surfaces (default: anchored)")
    p.add_argument('--refine', action='store_true')
    p.add_argument('--instance-policy', choices=('legacy', 'verified'), default='legacy')
    p.add_argument('--checkpoint-stages', action='store_true')
    p.add_argument('--resume-from', type=Path)
    args = p.parse_args()
    args.stride = 5
    def terminate(*_):
        raise KeyboardInterrupt('GUI cancelled')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    guard_gui_parent()
    from .semantic_runtime.pipeline import run
    args.gui_finalizes = True  # Complete once, after optional refinement.
    result = run(args)
    output = args.output.resolve()
    geom = read_json(output / 'mapping/mapping_result.json')
    final = result['map']
    classes = output / "fused/classes.json"
    if args.refine:
        root = prepare_refinement(output, args.manifest, args.runtime, args.vlm)
        cfg = read_json(args.runtime)
        # Same process group as bridge: GUI cancellation reaches active refinement.
        for stage, python in [('prepare', cfg['cpu_python']), ('name', cfg['vlm_python']),
                              ('decide', cfg['cpu_python']), ('ground', cfg['sam3_python']),
                              ('apply', cfg['cpu_python'])]:
            atomic_json(output / 'GUI_STAGE.json', {'stage': 'refine/' + stage})
            child = subprocess.Popen([python, '-m', 'pose_pipeline.semantic_runtime.refinement',
                                      '--workspace', str(root), '--stage', stage],
                                     env={**os.environ, 'REVEMAP_SUPERVISOR_PID': str(os.getpid())})
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
    raw_map = result['map']
    completion_receipt = None
    if args.map_completion == 'anchored':
        from .semantic_runtime.completion import publish
        atomic_json(output / 'GUI_STAGE.json', {'stage': 'completion'})
        final, completion_receipt = publish(output, args.manifest, final, output / 'completion')
        # The completed PLY carries labels and aligned RGB for the viewer.
        raw_map = final = str(final)
    atomic_json(output / 'GUI_STAGE.json', {'stage': 'export'})
    atomic_json(output / 'GUI_RESULT.json', {'status': 'completed', 'final_cloud': final, 'trajectory': geom['trajectory'],
                'classes': str(classes), 'raw_map': raw_map, 'base_map': result['map'],
                'map_completion': args.map_completion, 'names': result['names'], 'refinement': args.refine,
                'artifacts': str(output / 'ARTIFACTS.json'),
                'scope': 'run-sam3, optional unknown-point refinement, then selected map completion'})
    from .artifacts import write_artifact_manifest
    extras = {'names': result['names']}
    if completion_receipt:
        extras['completion'] = completion_receipt
    if geom.get('mesh'):
        from .surface import publish_surface
        surface = publish_surface(output, map_path=final, manifest=args.manifest,
                                  trajectory=geom['trajectory'],
                                  refusion_receipt=output / 'mapping/fusion/refusion_result.json')
        extras.update(surface=surface, mesh=geom['mesh'])
    write_artifact_manifest(output, map_path=Path(final), classes_path=classes,
                            result_path=output / 'GUI_RESULT.json', manifest_path=args.manifest,
                            trajectory_path=Path(geom['trajectory']),
                            # Existing IDs remain stable; promoted candidates
                            # have separate refinement naming evidence.
                            extra_files=extras)
    atomic_json(output / 'GUI_STAGE.json', {'stage': 'completed'})


if __name__ == '__main__':
    main()

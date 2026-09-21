"""Run quality-view refinement using explicit workspace and host runtimes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from ..common import read, write, sha, validate_runtime


def validate_workspace(root):
    scenes = read(root / 'INPUT_PLAN.json')['scenes']
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError('unique nonempty scenes required')
    files = [root / 'INPUT_PLAN.json', root / 'runtime.json']
    for scene in scenes:
        if Path(scene).is_absolute() or '..' in Path(scene).parts:
            raise ValueError('scene escapes workspace')
        inp = root / 'inputs' / scene
        files.extend(inp / n for n in ['INPUT.json', 'manifest.json', 'trajectory.json', 'target.npz', 'base.npz', 'classes.json'])
    return scenes, {str(p.relative_to(root)): sha(p) for p in files}


def run(root, stage, fragments=False):
    root = root.resolve(strict=True)
    scenes, inputs = validate_workspace(root)
    lock = root / 'REFINEMENT_INPUT_LOCK.json'
    if lock.exists():
        if read(lock) != inputs:
            raise ValueError('refinement inputs changed')
    else:
        write(lock, inputs)
    if stage == 'all':
        cfg = read(root / 'runtime.json')
        validate_runtime(cfg, 'qwen3vl_2b_nf4', raw_mapping=False)
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[3])
        env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
        for step, python in [('prepare', cfg['cpu_python']), ('name', cfg['vlm_python']),
                             ('decide', cfg['cpu_python']), ('ground', cfg['sam3_python']), ('apply', cfg['cpu_python'])]:
            cmd = [python, '-m', 'pose_pipeline.semantic_runtime.refinement', '--workspace', str(root), '--stage', step]
            if fragments:
                cmd.append('--fragments')
            subprocess.run(cmd, env=env, check=True)
        return
    from . import observations, grounding
    observations.R = grounding.R = root
    if stage == 'prepare':
        for scene in scenes:
            observations.prepare(scene)
    elif stage == 'name':
        observations.name_all(scenes)
    elif stage == 'decide':
        observations.decisions(scenes)
    elif stage == 'ground':
        grounding.ground()
    else:
        from .assignment import apply
        apply(root, fragments=fragments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'prepare', 'name', 'decide', 'ground', 'apply'], default='all')
    parser.add_argument('--fragments', action='store_true', help='opt-in three-view unknown-fragment policy')
    args = parser.parse_args()
    run(args.workspace, args.stage, args.fragments)


if __name__ == '__main__':
    main()

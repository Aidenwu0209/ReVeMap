"""Replay fusion with benchmark masks and fixed saved geometry in a new directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True,
                        help='Existing pipeline directory containing saved geometry and fused labels')
    parser.add_argument('--semantic', type=Path, required=True,
                        help='New benchmark SAM3 output directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'mapping').mkdir()
    shutil.copyfile(args.reference / 'mapping/mapping_result.json',
                    args.output / 'mapping/mapping_result.json')
    (args.output / 'semantic').symlink_to(args.semantic.resolve(), target_is_directory=True)
    from pose_pipeline.semantic_runtime.fusion import main as fuse
    fuse(argparse.Namespace(arm_root=args.output))
    with np.load(args.reference / 'fused/map_labels.npz') as x, np.load(args.output / 'fused/map_labels.npz') as y:
        equal = {key: x[key].dtype == y[key].dtype and np.array_equal(x[key], y[key]) for key in x.files}
    baseline_map = args.reference / 'fused/export/map_labeled.ply'
    output_map = args.output / 'fused/export/map_labeled.ply'
    report = {'scope': 'fixed original geometry plus fresh SAM3 evidence and fusion; no SLAM/VLM rerun',
              'reference': str(args.reference), 'semantic': str(args.semantic),
              'labels_equal': equal, 'map_sha256_before': digest(baseline_map),
              'map_sha256_after': digest(output_map)}
    report['passed'] = all(equal.values()) and report['map_sha256_before'] == report['map_sha256_after']
    (args.output / 'PARITY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

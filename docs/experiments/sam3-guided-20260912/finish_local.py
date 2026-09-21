"""Collect, preserve original geometry, render, then run unchanged fixed-GT evaluators."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

R = Path(__file__).resolve().parent
PREVIOUS = R.parent / 'sam3_sga_20260912_v1'
PYTHON = str(R.parent / 'sgf_sga_restore_20260910_v1/analysis_env/bin/python')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='consensus_instances')
    parser.add_argument('--scene', action='append', help='Repeat for selected keys; omitted means all five R2 scenes.')
    parser.add_argument('--skip-collect', action='store_true')
    args = parser.parse_args()
    jobs = json.loads((PREVIOUS / 'jobs_full_available.json').read_text()) + json.loads((PREVIOUS / 'jobs_3rscan_archived.json').read_text())
    keys = args.scene or [j['key'] for j in jobs]
    if not set(keys) <= {j['key'] for j in jobs}:
        raise ValueError('Requested scene is outside the frozen five-scene set.')
    for key in keys:
        common = ['--stage', args.stage, '--scene', key]
        if not args.skip_collect:
            subprocess.run([sys.executable, str(R / 'collect_results.py')] + common, check=True)
        run = R / args.stage / key
        if not (run / 'map').exists():
            export_flags = ['--allow-unknown-instances'] if args.stage in ('unknown_contract', 'raw_unknown') else []
            subprocess.run([PYTHON, str(R / 'export_result.py')] + common + export_flags, check=True, stdout=subprocess.DEVNULL)
        else:
            contract = json.loads((run / 'map/MAP_CONTRACT.json').read_text())
            import hashlib
            if hashlib.sha256((run / 'map_labels.npz').read_bytes()).hexdigest() != contract['label_sha256']:
                raise RuntimeError('Existing map belongs to different labels; use a fresh stage.')
        subprocess.run([PYTHON, str(R / 'render_maps.py')] + common, check=True)
        if key == 'scannet/scene0030_00':
            path = str(run / 'map/map_labeled.ply')
            for script, flag, output in [('evaluate_0030.py', '--sam-map', f'SEMANTIC_0030_{args.stage}.json'),
                                         ('evaluate_instances_0030.py', '--map', f'INSTANCE_0030_{args.stage}.json')]:
                target = R / output
                if not target.exists():
                    subprocess.run([PYTHON, str(R / script), flag, path, '--output', str(target)], check=True)
            secondary = R / f'CLASSAGNOSTIC_0030_{args.stage}.json'
            if not secondary.exists():
                subprocess.run([PYTHON, str(R / 'evaluate_classagnostic_0030.py'),
                                '--labels', str(run / 'map_labels.npz'), '--output', str(secondary)], check=True)
        if key.startswith('orbbec/') and not (run / 'camera_audit').exists():
            subprocess.run([PYTHON, str(R / 'render_orbbec_comparison.py'), '--sam-labels',
                            str(run / 'map_labels.npz'), '--output', str(run / 'camera_audit')], check=True)
    print('LOCAL_COLLECTION_EXPORT_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

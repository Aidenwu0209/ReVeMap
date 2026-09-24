"""Check exact semantic evidence parity between two SAM3 benchmark directories."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(left, right):
    def read(root, name):
        return json.loads((root / name).read_text())
    a, b = read(left, 'FRAMES.json'), read(right, 'FRAMES.json')
    errors = []
    ids = [row['frame_id'] for row in a]
    if ids != [row['frame_id'] for row in b]:
        errors.append('selected frame IDs differ')
    checked = 0
    for ar, br in zip(a, b):
        for key in ('frame_id', 'color_sha256', 'depth_sha256', 'crops'):
            if ar[key] != br[key]:
                errors.append(f'{ar["frame_id"]}: {key}')
        filename = f'{ar["frame_id"]:06}.npz'
        with np.load(left / 'frames' / filename) as x, np.load(right / 'frames' / filename) as y:
            if set(x.files) != set(y.files):
                errors.append(filename + ': array keys')
            for key in x.files:
                checked += 1
                if key not in y or x[key].dtype != y[key].dtype or not np.array_equal(x[key], y[key]):
                    errors.append(filename + ': ' + key)
    def normalize(tasks):
        for task in tasks:
            for crop in task['crops']:
                crop['file'] = Path(crop['file']).name
        return tasks
    if normalize(read(left, 'CROP_TASKS.json')) != normalize(read(right, 'CROP_TASKS.json')):
        errors.append('crop tasks differ')
    crops_a = sorted(p.name for p in (left / 'crops').glob('*.png'))
    crops_b = sorted(p.name for p in (right / 'crops').glob('*.png'))
    if crops_a != crops_b:
        errors.append('crop file set differs')
    for name in crops_a:
        if not (right / 'crops' / name).exists() or (left / 'crops' / name).read_bytes() != (right / 'crops' / name).read_bytes():
            errors.append('crop bytes: ' + name)
    for key in ('checkpoint_sha256', 'missing_keys', 'unexpected_keys', 'precision', 'seed'):
        if read(left, 'MODEL.json')[key] != read(right, 'MODEL.json')[key]:
            errors.append('model: ' + key)
    return {'scope': 'exact saved masks, semantic, instance, confidence, input digests and crop evidence',
            'left': str(left), 'right': str(right), 'frames': len(ids),
            'arrays_checked': checked, 'crops_checked': len(crops_a), 'errors': errors,
            'passed': not errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)

"""Attach experimentally inferred labels while preserving every original vertex."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .contracts import sha256_file


def export(baseline, labels, result, output):
    from plyfile import PlyData, PlyElement
    source = PlyData.read(baseline)
    v = source['vertex'].data
    xyz = np.stack([v[k] for k in ('x', 'y', 'z')], axis=1)
    report = json.loads(result.read_text())
    xyz_hash = hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()
    if report['status'] != 'completed' or report.get('geometry_xyz_sha256') != xyz_hash:
        raise ValueError('completed label output must match exact baseline point order and precision')
    with np.load(labels) as d:
        sem, inst, conf = (d[k].copy() for k in ('semantic', 'instance', 'confidence'))
    if any(x.shape != (len(v),) for x in (sem, inst, conf)):
        raise ValueError('label count mismatch')
    if np.any(sem < 0) or np.any(inst < 0) or not np.isfinite(conf).all():
        raise ValueError('invalid labels')
    if np.any((inst > 0) & (sem == 0)):
        raise ValueError('instance cannot have unknown category')
    fields = [('semantic_id', '<i4'), ('instance_id', '<i4'), ('semantic_confidence', '<f4')]
    if any(k in v.dtype.names for k, _ in fields):
        raise ValueError('baseline already contains semantic fields; use original geometry')
    original_sha = sha256_file(baseline)
    output.mkdir(parents=True, exist_ok=False)
    labeled = np.empty(len(v), dtype=v.dtype.descr + fields)
    for key in v.dtype.names:
        labeled[key] = v[key]
    labeled['semantic_id'], labeled['instance_id'], labeled['semantic_confidence'] = sem, inst, conf

    def write(path, vertices):
        elements = [PlyElement.describe(vertices, 'vertex') if e.name == 'vertex' else e for e in source.elements]
        PlyData(elements, text=False).write(str(path))

    write(output/'map_labeled.ply', labeled)
    check = PlyData.read(output/'map_labeled.ply')['vertex'].data
    preserved = {k: np.array_equal(check[k], v[k], equal_nan=True) for k in v.dtype.names}
    if not all(preserved.values()) or sha256_file(baseline) != original_sha:
        raise RuntimeError('original map preservation failed')
    for key in ('semantic_id', 'instance_id'):
        colored = labeled.copy(); ids = labeled[key].astype(np.int64)
        for channel, multiplier in zip(('red', 'green', 'blue'), (73, 151, 199)):
            colored[channel] = np.where(ids > 0, 50 + ids * multiplier % 206, 90).astype(np.uint8)
        write(output/f'map_{key}.ply', colored)
    np.save(output/'semantic.npy', sem); np.save(output/'instance.npy', inst)
    receipt = {'baseline_sha256': original_sha, 'geometry_xyz_sha256': xyz_hash,
               'original_vertex_properties_preserved': preserved, 'point_count': len(v),
               'semantic_coverage': float(np.mean(sem > 0)), 'instance_coverage': float(np.mean(inst > 0)),
               'inference_result': str(result), 'label_sha256': sha256_file(labels),
               'sga_inference_executed': report['sga_inference_executed'],
               'quality_accepted': False, 'complete_full_sequence': report['complete_full_sequence']}
    (output/'MAP_CONTRACT.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'labels', 'result', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.baseline, args.labels, args.result, args.output), indent=2))


if __name__ == '__main__':
    main()

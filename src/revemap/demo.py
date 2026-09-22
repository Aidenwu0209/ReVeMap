"""Small, explicitly synthetic CPU example for artifact and query workflows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def create_demo(output):
    from plyfile import PlyData, PlyElement
    from pose_pipeline.artifacts import write_artifact_manifest
    from pose_pipeline.scene_graph import build_from_artifacts

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    boxes = [([0, 0, .7], [1.2, .8, .8], 1),
             ([.4, .25, .82], [.55, .4, 1.05], 2),
             ([1.6, .1, 0], [2.1, .6, .9], 3)]
    rows = []
    for low, high, oid in boxes:
        points = np.array(np.meshgrid(*[np.linspace(a, b, 6) for a, b in zip(low, high)])).reshape(3, -1).T
        surface = np.any(np.isclose(points, low) | np.isclose(points, high), axis=1)
        rows.extend([(*p, 90 + oid * 35, 140, 180, oid, oid, .9) for p in points[surface]])
    vertices = np.array(rows, dtype=[(k, '<f4') for k in ('x', 'y', 'z')]
                        + [(k, 'u1') for k in ('red', 'green', 'blue')]
                        + [('semantic_id', '<i4'), ('instance_id', '<i4'), ('semantic_confidence', '<f4')])
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(output / 'map_labeled.ply')
    (output / 'classes.json').write_text(json.dumps({'0': 'unknown', '1': 'table', '2': 'cup', '3': 'chair'}, indent=2) + '\n')
    (output / 'result.json').write_text(json.dumps({
        'status': 'completed', 'data_source': 'synthetic_geometric_boxes',
        'model_inference_executed': False, 'scope': 'CPU artifact/query demonstration; not mapping or accuracy evidence',
    }, indent=2) + '\n')
    artifact = write_artifact_manifest(output, map_path=output / 'map_labeled.ply',
                                      classes_path=output / 'classes.json', result_path=output / 'result.json')
    graph = build_from_artifacts(artifact, world_up=[0, 0, 1])
    (output / 'scene_graph.json').write_text(json.dumps(graph, indent=2, allow_nan=False) + '\n')
    return {'output': str(output), 'artifacts': str(artifact), 'objects': len(graph['nodes']),
            'synthetic': True, 'model_inference_executed': False}


def add_command(commands):
    command = commands.add_parser('demo', help='create a synthetic CPU example; no model inference')
    command.add_argument('--output', type=Path, required=True)
    command.set_defaults(handler=lambda args: print(json.dumps(create_demo(args.output), indent=2)))

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pose_pipeline.artifacts import write_artifact_manifest
from pose_pipeline.contracts import sha256_file
from pose_pipeline.live_gui import Controller
from revemap.demo import create_demo


def completed_controller(tmp_path):
    session = tmp_path / 'sessions' / 'synthetic'
    pipeline = session / 'pipeline'
    create_demo(pipeline)
    controller = Controller(SimpleNamespace(replay=None, output=session.parent))
    controller.session = session
    controller.state = {'status': 'completed', 'pipeline': str(pipeline), 'result': {
        'final_cloud': str(pipeline / 'map_labeled.ply'), 'raw_map': str(pipeline / 'map_labeled.ply'),
        'classes': str(pipeline / 'classes.json'), 'artifacts': str(pipeline / 'ARTIFACTS.json')}}
    return controller, pipeline


def test_query_highlights_real_instance_and_clear_restores_color(tmp_path):
    controller, pipeline = completed_controller(tmp_path)
    result = controller.query_scene({'label': '椅子'})
    assert result['query_result']['instance_ids'] == [3]
    assert result['state']['highlighted_instances'] == [3]
    meta = json.loads((controller.session / 'cloud.json').read_text())
    points = np.fromfile(controller.session / meta['file'], dtype='<f4').reshape(-1, 6)
    assert np.any(np.all(np.isclose(points[:, 3:], [1, .85, .25]), axis=1))
    controller.show_cloud('rgb')
    assert controller.state['highlighted_instances'] is None
    assert controller.query_scene({'relation': 'above', 'reference_id': 1})['query_result']['reason'] == 'world_up_required'
    assert controller.query_scene({'relation': 'supported_by', 'reference_id': 1, 'world_up': [0, 0, 1]})['query_result']['instance_ids'] == [2]
    assert controller.query_scene({'label': 'sofa'})['query_result']['status'] == 'unknown'
    with pytest.raises(ValueError):
        controller.query_scene({'nearest_to': -1})
    with pytest.raises(ValueError):
        controller.show_cloud('rgb', [999])


def test_graph_cache_does_not_hide_modified_class_file(tmp_path):
    controller, pipeline = completed_controller(tmp_path)
    controller.query_scene({'label': 'chair'})
    (pipeline / 'classes.json').write_text('{"0":"unknown","1":"altered"}')
    with pytest.raises(ValueError, match='digest mismatch'):
        controller.query_scene({'label': 'chair'})


def test_only_registered_hash_checked_crop_can_be_served(tmp_path):
    controller, pipeline = completed_controller(tmp_path)
    crop = pipeline / 'crop.png'
    Image.new('RGB', (8, 8), 'blue').save(crop)
    names = pipeline / 'names.json'
    names.write_text(json.dumps([{'instance_id': 3, 'vlm_name': 'stool', 'support_frames': [1, 2],
                                 'evidence': [{'frame_id': 1, 'file': str(crop), 'sha256': sha256_file(crop)}]}]))
    inventory = write_artifact_manifest(pipeline / 'with-names', map_path=pipeline / 'map_labeled.ply',
        classes_path=pipeline / 'classes.json', result_path=pipeline / 'result.json', extra_files={'names': names})
    controller.state['result']['artifacts'] = str(inventory)
    assert controller.query_scene({'label': 'stool'})['query_result']['instance_ids'] == [3]
    context = controller.scene_context()
    assert controller.evidence_image(3, 0, context).startswith(b'\xff\xd8')
    with pytest.raises(ValueError):
        controller.evidence_image(3, -1, context)
    Image.new('RGB', (8, 8), 'red').save(crop)
    with pytest.raises(ValueError, match='改变'):
        controller.evidence_image(3, 0, context)


def test_stale_tab_cannot_highlight_or_fetch_another_attempts_evidence(tmp_path):
    controller, _ = completed_controller(tmp_path)
    context = controller.query_scene({'label': 'chair'})['query_result']['context']
    controller.state['attempt'] = str(tmp_path / 'different-attempt')
    with pytest.raises(ValueError, match='切换'):
        controller.query_scene({'label': 'chair', 'context': context})
    with pytest.raises(ValueError, match='切换'):
        controller.show_cloud('semantic_id', [3], context)
    with pytest.raises(ValueError, match='切换'):
        controller.evidence_image(3, 0, context)


def test_small_highlight_survives_preview_decimation(tmp_path):
    from plyfile import PlyData, PlyElement
    controller, pipeline = completed_controller(tmp_path)
    original = PlyData.read(pipeline / 'map_labeled.ply')['vertex'].data
    vertices = np.repeat(original[:1], 100001)
    vertices['instance_id'] = 0
    vertices['instance_id'][99999] = 7
    vertices['x'][99999] = 123.
    cloud = pipeline / 'large.ply'
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(cloud)
    controller.state['result']['final_cloud'] = str(cloud)
    controller.show_cloud('semantic_id', [7], controller.scene_context())
    meta = json.loads((controller.session / 'cloud.json').read_text())
    preview = np.fromfile(controller.session / meta['file'], dtype='<f4').reshape(-1, 6)
    assert len(preview) <= 100000
    assert np.any((preview[:, 0] == 123) & np.all(np.isclose(preview[:, 3:], [1, .85, .25]), axis=1))

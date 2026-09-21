import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from pose_pipeline.live_gui import Controller
from pose_pipeline.live_semantic import prepare_refinement
from pose_pipeline.live_io import publish_cloud


def test_start_rejects_bad_options_before_capture(tmp_path):
    args = SimpleNamespace(replay=None, runtime=tmp_path/'runtime.json')
    c = Controller(args)
    for options in ({'vlm': 'not-a-model'}, {'schedule': 'anything'}, {'refine': 'true'}):
        with pytest.raises(ValueError):
            c.start(options)
    assert c.thread is None and c.session is None


def test_bridge_preserves_arrays_and_provenance(tmp_path):
    out = tmp_path/'pipeline'; (out/'mapping').mkdir(parents=True); (out/'fused').mkdir()
    manifest=tmp_path/'manifest.json'; manifest.write_text('{}')
    runtime=tmp_path/'runtime.json'; runtime.write_text('{}')
    trajectory=tmp_path/'trajectory.json'; trajectory.write_text('{"estimated":true}')
    (out/'mapping/mapping_result.json').write_text(json.dumps({'trajectory':str(trajectory)}))
    np.savez(out/'fused/target.npz', xyz=np.array([[1,2,3],[4,5,6]]))
    np.savez(out/'fused/map_labels.npz', semantic=[0,1], instance=[3,4], confidence=[0.,.9])
    (out/'fused/classes.json').write_text('{"0":"unknown","1":"chair"}')
    root=prepare_refinement(out,manifest,runtime)
    assert (root/'inputs/capture/base.npz').read_bytes()==(out/'fused/map_labels.npz').read_bytes()
    assert json.loads((root/'inputs/capture/INPUT.json').read_text())['rgb_registration']=='already_registered'
    with pytest.raises(FileExistsError): prepare_refinement(out,manifest,runtime)


def test_cloud_generation_and_nan_filter(tmp_path):
    publish_cloud(tmp_path,np.array([[1,2,3,1,0,0],[np.nan,0,0,0,0,0]]),kind='final',revision=1)
    meta=json.loads((tmp_path/'cloud.json').read_text())
    assert meta['points']==1
    assert np.fromfile(tmp_path/meta['file'],dtype='<f4').size==6


def test_show_cloud_semantic_instance_rgb(tmp_path):
    from plyfile import PlyData,PlyElement
    vertices=np.zeros(2,dtype=[(x,'f4') for x in ['x','y','z']]+[(x,'u1') for x in ['red','green','blue']]+[(x,'i4') for x in ['semantic_id','instance_id']])
    vertices['semantic_id']=[0,1]; vertices['instance_id']=[0,9]; vertices['red']=255
    ply=tmp_path/'map.ply'; PlyData([PlyElement.describe(vertices,'vertex')]).write(ply)
    c=Controller(SimpleNamespace(replay=None)); c.session=tmp_path
    c.state={'status':'completed','result':{'final_cloud':str(ply),'raw_map':str(ply)}}
    for mode in ['semantic_id','instance_id','rgb']:
        c.show_cloud(mode)
        meta=json.loads((tmp_path/'cloud.json').read_text()); v=np.fromfile(tmp_path/meta['file'],dtype='<f4').reshape(-1,6)
        assert v.shape==(2,6) and np.isfinite(v).all()
    assert np.array_equal(v[:,3],[1,1])
    with pytest.raises(ValueError): c.show_cloud('garbage')

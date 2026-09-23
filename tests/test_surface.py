import json
import os
import struct

import numpy as np
import pytest

from pose_pipeline.contracts import sha256_file
from pose_pipeline.surface import SCHEMA, encode_mesh
from test_device_queries import server  # shared real HTTP fixture


def publish_fixture(session):
    pipeline = session / 'pipeline'
    result = json.loads((pipeline / 'GUI_RESULT.json').read_text())
    mesh = pipeline / 'surface.ply'
    mesh.write_bytes(b'ply\nformat ascii 1.0\nelement vertex 3\nelement face 1\nend_header\n')
    manifest = pipeline / 'input.json'
    manifest.write_text('{}')
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    blob = encode_mesh(vertices, np.ones((3,3)), np.tile([0,0,1.], (3,1)), [[0,1,2]])
    viewer = pipeline / 'surface.bin'; viewer.write_bytes(blob)
    receipt = pipeline / 'surface-receipt.json'
    receipt.write_text(json.dumps(dict(status='completed', mesh_sha256=sha256_file(mesh),
        manifest_sha256=sha256_file(manifest), trajectory_sha256=sha256_file(result['trajectory']))))
    paths = dict(map=result['final_cloud'], trajectory=result['trajectory'],
                 input_manifest=manifest, mesh=mesh, refusion=receipt, viewer=viewer)
    index = dict(schema=SCHEMA, vertices=3, triangles=1, viewer_vertices=3, viewer_triangles=1,
        files={k:dict(path=os.path.relpath(v,pipeline), sha256=sha256_file(v)) for k,v in paths.items()})
    (pipeline / 'SURFACE.json').write_text(json.dumps(index))
    return blob, viewer


def test_surface_encoding_preserves_triangles_and_rejects_invalid_geometry():
    vertices = np.eye(3)
    blob = encode_mesh(vertices, vertices, vertices, [[0,1,2]])
    assert blob[:8] == b'RVMESH01'
    assert struct.unpack('<II',blob[8:16]) == (3,1)
    assert np.frombuffer(blob[-12:],dtype='<u4').tolist() == [0,1,2]
    with pytest.raises(ValueError): encode_mesh(vertices,vertices,vertices,[[0,1,3]])
    with pytest.raises(ValueError): encode_mesh(vertices,vertices,vertices,[[0.,1.,2.]])
    vertices[0,0] = np.nan
    with pytest.raises(ValueError): encode_mesh(vertices,vertices,vertices,[[0,1,2]])


def test_surface_http_is_session_bound_and_does_not_rewrite_map(server):
    controller,request,first,second=server
    old={p:p.read_bytes() for p in second.rglob('*') if p.is_file()}
    blob,_=publish_fixture(second)
    base='/s/'+second.name
    code,meta=request(base+'/api/surface')
    assert code==200 and meta['available'] and meta['triangles']==1
    query='?revision='+meta['revision']+'&context='+meta['context']
    assert request(base+'/surface.bin'+query)==(200,blob)
    assert request(base+'/surface.ply'+query)[1].startswith(b'ply\n')
    assert request(base+'/surface.bin?revision=old')[0]==409
    assert request(base+'/surface.ply?context=other-map')[0]==409
    assert request('/s/'+first.name+'/api/surface')[1]['available'] is False
    assert request('/s/'+first.name+'/surface.ply')[0]==404
    assert controller.session is None
    assert all(p.read_bytes()==data for p,data in old.items())


def test_changed_or_cross_session_surface_is_rejected(server):
    _,request,first,second=server
    _,viewer=publish_fixture(second)
    base='/s/'+second.name
    viewer.write_bytes(b'changed')
    assert request(base+'/surface.bin')[0]==409
    index=second/'pipeline/SURFACE.json'
    data=json.loads(index.read_text())
    data['files']['viewer']={'path':str(first/'session.json'),'sha256':sha256_file(first/'session.json')}
    index.write_text(json.dumps(data))
    assert request(base+'/surface.bin')[0]==409


def test_surface_of_failed_session_is_not_exposed(server):
    _,request,_,second=server
    publish_fixture(second)
    state=second/'session.json';data=json.loads(state.read_text());data['status']='failed';state.write_text(json.dumps(data))
    assert request('/s/'+second.name+'/api/surface')[0]==409

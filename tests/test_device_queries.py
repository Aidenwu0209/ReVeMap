import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from pose_pipeline.device_gui import Controller, handler, completed_result
from pose_pipeline.live_io import atomic_json
from revemap.demo import create_demo


def saved_scene(root, sid):
    session=root/sid;pipeline=session/'pipeline';create_demo(pipeline)
    (pipeline/'trajectory.json').write_text('{}')
    result={'final_cloud':str(pipeline/'map_labeled.ply'),'raw_map':str(pipeline/'map_labeled.ply'),
            'classes':str(pipeline/'classes.json'),'trajectory':str(pipeline/'trajectory.json')}
    atomic_json(pipeline/'GUI_RESULT.json',result)
    atomic_json(session/'session.json',{'status':'completed','mode':'demo','result':result})
    return session


@pytest.fixture
def server(tmp_path):
    primary=tmp_path/'mobile';extra=tmp_path/'wireless'
    first=saved_scene(primary,'scan_20260923_111111_a1')
    second=saved_scene(extra,'scan_20260923_222222_a2')
    args=SimpleNamespace(replay=None,output=primary,library_root=[extra],ipad_port=0,wireless_host=None)
    controller=Controller(args)
    http=ThreadingHTTPServer(('127.0.0.1',0),handler(controller,'test-token',0))
    worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
    def request(path,payload=None,token='test-token'):
        req=Request(f'http://127.0.0.1:{http.server_port}'+path,
                    data=json.dumps(payload).encode() if payload is not None else None,
                    headers={'X-Scan-Token':token,'Content-Type':'application/json'})
        try:
            with urlopen(req,timeout=4) as response:
                data=response.read();mime=response.headers.get_content_type()
                return response.status,json.loads(data) if mime=='application/json' else data
        except HTTPError as error:return error.code,json.loads(error.read())
    yield controller,request,first,second
    http.shutdown();http.server_close();worker.join()


def test_query_saved_wireless_scene_without_switching_active_scan(server):
    controller,request,first,second=server
    assert len(request('/api/sessions')[1]['sessions'])==2
    base='/s/'+second.name
    status,state=request(base+'/api/status');assert status==200
    original={p:p.read_bytes() for p in second.rglob('*') if p.is_file()}
    status,result=request(base+'/api/query',{'question':'有几把椅子','context':state['scene_context']})
    assert status==200 and result['query_result']['instance_ids']==[3]
    assert result['query_result']['count']==1
    assert controller.session is None
    assert all(p.read_bytes()==data for p,data in original.items())
    assert request(base+'/scene_graph.json')[1]['nodes']
    assert request(base+'/api/objects')[1]['points']>0
    assert request(base+'/object.ply?kind=instance&id=3')[0]==200
    assert request(base+'/trajectory.json')[0]==200


def test_context_is_required_and_cannot_cross_sessions(server):
    _,request,first,second=server
    one=request('/s/'+first.name+'/api/status')[1]['scene_context']
    base='/s/'+second.name+'/api/query'
    assert request(base,{'question':'椅子','context':one})[0]==409
    assert request(base,{'question':'椅子'})[0]==409
    assert request(base,{'question':'椅子'},token='wrong')[0]==403


def test_failed_session_does_not_expose_old_completed_map(server):
    _,request,first,_=server
    old=json.loads((first/'session.json').read_text());old['status']='failed';atomic_json(first/'session.json',old)
    base='/s/'+first.name
    assert request(base+'/api/status')[1]['status']=='failed'
    assert request(base+'/api/graph')[0]==409
    assert request(base+'/api/objects')[0]!=200
    assert request(base+'/final.ply')[0]!=200
    assert request(base+'/trajectory.json')[0]!=200
    with pytest.raises(ValueError,match='not completed'):completed_result(first)


def test_corrupt_paths_and_unknown_session_do_not_break_server(server):
    _,request,first,_=server
    assert request('/s/scan_20260923_999999_x/api/graph')[0]==409
    assert request('/s/../../private/api/graph')[0] in (404,409)
    assert request('/api/sessions')[0]==200
    result=json.loads((first/'pipeline/GUI_RESULT.json').read_text())
    result['final_cloud']='/etc/passwd';atomic_json(first/'pipeline/GUI_RESULT.json',result)
    assert request('/s/'+first.name+'/api/graph')[0]==409


def test_session_library_rejects_duplicate_ids(server):
    controller,_,first,second=server
    saved_scene(second.parent,first.name)
    with pytest.raises(ValueError,match='ambiguous'):controller.resolve_session(first.name)

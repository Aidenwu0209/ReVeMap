"""Terminal ownership and interface checks; these fixtures are not quality data."""
import hashlib
import json
from pathlib import Path
from argparse import Namespace
import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from pose_pipeline.contracts import FrameRecord, PoseRecord, SequenceManifest, write_manifest, write_trajectory
from pose_pipeline.semantic_runtime.geometry_readout import apply_geometry_readout
from pose_pipeline.semantic_runtime.geometry_contract import save_contract, validate_contract, artifact_files, xyz_hash
from pose_pipeline.semantic_runtime.multiview import fuse_instances
from pose_pipeline.sam3_export import export
from pose_pipeline.scene_graph import build_graph, build_from_artifacts, query_graph
from pose_pipeline.artifacts import load_artifacts
from pose_pipeline.semantic_runtime.backfill import main as backfill


def labels(sem, inst):
    return {'semantic': np.asarray(sem, np.int32), 'instance': np.asarray(inst, np.int32),
            'confidence': np.where(np.asarray(sem) > 0, .9, 0).astype(np.float32)}


def prepared(tmp_path, *, multi_owner=False):
    root = tmp_path / 'fused'; root.mkdir()
    base = labels([1, 1, 0, 0, 0, 0, 0, 2, 2], [1, 1, 0, 0, 0, 0, 0, 2, 2])
    safe = base['instance'].copy(); geometry = np.array([1]*7 + [2]*2, np.int32)
    if multi_owner:
        geometry[-2:] = 1
    capture = {'object_id': geometry, 'safe_current': safe, 'eligible_group_ids': np.arange(1, int(geometry.max())+1, dtype=np.int32)}
    final, audit, lineage = apply_geometry_readout(base, base, safe, geometry)
    xyz = np.column_stack([np.arange(9)*.1, np.zeros(9), np.ones(9)])
    raw = tmp_path / 'raw'; raw.mkdir(); color = raw/'color.jpg'; depth = raw/'depth.png'
    color.write_bytes(b'rgb'); depth.write_bytes(b'depth')
    manifest = tmp_path/'input.json'; trajectory = tmp_path/'trajectory.json'
    sequence = SequenceManifest('scannet','scene0062_00',raw,1000.,(FrameRecord(0,0,color,depth,(1.,1.,0.,0.)),),'synthetic-interface')
    write_manifest(manifest,sequence);write_trajectory(trajectory,[PoseRecord(0,0,np.eye(4))],sequence_id=sequence.sequence_id,arm='synthetic')
    provenance = {'manifest':str(manifest),'trajectory':str(trajectory),'scene_id':sequence.sequence_id,'dataset':'scannet'}
    np.savez_compressed(root/'map_labels.npz', **final);np.savez_compressed(root/'target.npz',xyz=xyz)
    contract=save_contract(root,xyz,base,final,capture,audit,lineage,provenance)
    report={'status':'completed','geometry_xyz_sha256':xyz_hash(xyz),'sga_inference_executed':False,
            'complete_full_sequence':False,'geometry_readout':'owner-lineage-v1','provenance':provenance}
    (root/'result.json').write_text(json.dumps(report));(root/'classes.json').write_text(json.dumps({'0':'unknown','1':'chair','2':'table'}))
    verts=np.zeros(len(xyz),dtype=[(k,'f8') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')])
    for i,k in enumerate(('x','y','z')):verts[k]=xyz[:,i]
    cloud=tmp_path/'raw.ply';PlyData([PlyElement.describe(verts,'vertex')]).write(str(cloud))
    return root,base,final,xyz,contract,cloud


def rebind(root, filename):
    p=root/'GEOMETRY_CONTRACT.json';c=json.loads(p.read_text())
    if filename=='map_labels.npz':
        c['final_labels']['sha256']=hashlib.sha256((root/filename).read_bytes()).hexdigest()
        with np.load(root/filename) as z:
            for k in z.files:c['label_arrays'][k]={'dtype':z[k].dtype.str,'shape':list(z[k].shape),'sha256':hashlib.sha256(z[k].tobytes()).hexdigest()}
    else:
        for entry in c['files'].values():
            if entry['path']==filename:entry['sha256']=hashlib.sha256((root/filename).read_bytes()).hexdigest()
    p.write_text(json.dumps(c))


def test_unknown_majority_preserves_evidence_category_and_expands_geometry(tmp_path):
    root,base,final,xyz,contract,cloud=prepared(tmp_path)
    old=build_graph(xyz,base['semantic'],base['instance'],{'0':'unknown','1':'chair','2':'table'},confidence=base['confidence'])
    checked=validate_contract(contract,xyz,final);assert checked['audit']['added_points']==5
    export(cloud,root/'map_labels.npz',root/'result.json',root/'export',geometry_contract=contract)
    graph=build_from_artifacts(root/'export')
    chair=graph['nodes'][0];assert chair['semantic_id']==1 and chair['label']=='chair'
    assert chair['semantic_class_counts']==old['nodes'][0]['semantic_class_counts']
    assert chair['mean_observation_score']==old['nodes'][0]['mean_observation_score']
    assert chair['point_count']==7 and chair['unknown_geometry_points']==5
    assert chair['aabb_max_m'][0]==pytest.approx(.6)
    assert query_graph(graph,label='chair')['instance_ids']==[1]
    assert np.array_equal(final['semantic'],base['semantic'])
    assert len(load_artifacts(root/'export')['paths'])>=10


def test_export_requires_explicit_contract_before_creating_output(tmp_path):
    root,base,final,xyz,contract,cloud=prepared(tmp_path)
    with pytest.raises(ValueError,match='explicit geometry contract'):
        export(cloud,root/'map_labels.npz',root/'result.json',root/'bad')
    assert not (root/'bad').exists()
    with pytest.raises(ValueError,match='point order'):
        validate_contract(contract,xyz[::-1].copy(),final)


def test_multiple_safe_owners_abstain_even_if_one_group_has_unknown_candidates(tmp_path):
    root,base,final,xyz,contract,cloud=prepared(tmp_path,multi_owner=True)
    assert np.array_equal(base['instance'],final['instance'])
    check=validate_contract(contract,xyz,final)
    assert check['audit']['links'][0]['reason']=='multiple_original_safe_output_owners'
    assert check['audit']['added_points']==0


@pytest.mark.parametrize('mutation',['semantic','confidence','old_owner','invented_owner','extra_point','lineage'])
def test_forged_changes_rejected_even_after_receipt_hash_rebinding(tmp_path,mutation):
    root,base,final,xyz,contract,cloud=prepared(tmp_path)
    if mutation=='lineage':
        p=root/'GEOMETRY_POINT_LINEAGE.npz'
        with np.load(p) as z:d={k:z[k].copy() for k in z.files}
        d['final_owner_id'][0]=2;np.savez_compressed(p,**d);rebind(root,p.name)
    else:
        if mutation=='semantic':final['semantic'][2]=1
        elif mutation=='confidence':final['confidence'][2]=.5
        elif mutation=='old_owner':final['instance'][0]=2
        elif mutation=='invented_owner':final['instance'][2]=77
        elif mutation=='extra_point':final['instance'][8]=1
        np.savez_compressed(root/'map_labels.npz',**final);rebind(root,'map_labels.npz')
    with pytest.raises(ValueError,match='exact terminal readout'):
        validate_contract(contract,xyz,final)


def test_source_mutation_and_missing_artifact_sidecar_rejected(tmp_path):
    root,base,final,xyz,contract,cloud=prepared(tmp_path)
    files=artifact_files(root);files.pop('geometry_lineage')
    with pytest.raises(ValueError,match='bind every'):
        validate_contract(contract,xyz,final,artifact_paths=files)
    (tmp_path/'trajectory.json').write_text('{}')
    with pytest.raises(ValueError,match='provenance digest'):
        validate_contract(contract,xyz,final)


def test_same_call_ledger_capture_preserves_output_and_audit_and_multiclass_split():
    n=120; ids=np.arange(n,dtype=np.int32)
    frames=[{'frame_id':f,'point_ids':ids,'mask_ids':np.ones(n,np.int32),'semantic':np.ones(n,np.int32),
             'confidence':np.full(n,.9,np.float32),'interior':np.ones(n,bool)} for f in range(5)]
    sem=np.r_[np.ones(50,int),np.full(50,2),np.zeros(20,int)].astype(np.int32)
    cfg={'min_mask_points':30,'min_output_points':50,'min_point_views':1,'min_group_frames':2,'object_score_mode':'max_point'}
    before,audit=fuse_instances(n,frames,sem,cfg);capture={}
    after,also=fuse_instances(n,frames,sem,cfg,geometry_out=capture)
    assert np.array_equal(before,after) and audit==also
    assert np.unique(capture['object_id']).tolist()==[1]
    assert np.unique(after).tolist()==[0,1,2]
    base=labels(sem,after);result,check,_=apply_geometry_readout(base,base,capture['safe_current'],capture['object_id'])
    assert not result['instance'][-20:].any()
    assert check['links'][0]['reason']=='multiple_original_safe_output_owners'
    empty={};fuse_instances(n,[],sem,cfg,geometry_out=empty)
    assert set(empty)==set(capture) and not empty['object_id'].any()


def test_backfill_uses_pre_readout_crop_purity_and_preserves_names(tmp_path):
    root,base,final,xyz,contract,cloud=prepared(tmp_path)
    vlm=tmp_path/'semantic/vlm';vlm.mkdir(parents=True)
    # Repeat sparse observed points to satisfy the existing count gate. Unknown
    # additions change final crop purity from 60/130 to 130/130; naming must
    # retain the original rejection (purity .46), with no new model calls.
    points=np.r_[np.tile([0,1],30),np.tile([2,3,4,5,6],14)]
    for f in [0,1]:
        np.savez_compressed(root/('projection_%06d.npz'%f),point_ids=points,mask_ids=np.ones(len(points),int))
        (vlm/('response_%06d.json'%f)).write_text(json.dumps({'completed_at':1,'crops':[{'frame_id':f,'mask_id':1,'label':'chair'}]}))
    backfill(Namespace(arm_root=tmp_path));on=json.loads((root/'instance_names.json').read_text())
    assert on[0]['vlm_name']=='unknown' and on[0]['semantic_id']==1 and on[0]['point_count']==7
    rejected=json.loads((root/'NAME_REJECTIONS.json').read_text());assert len(rejected)==2
    report=json.loads((root/'result.json').read_text());report.pop('geometry_readout');(root/'result.json').write_text(json.dumps(report))
    np.savez_compressed(root/'map_labels.npz',**base)
    backfill(Namespace(arm_root=tmp_path));off=json.loads((root/'instance_names.json').read_text())
    assert on[0]['evidence']==off[0]['evidence'] and on[0]['support_frames']==off[0]['support_frames']
    assert on[0]['vlm_name']==off[0]['vlm_name']
    assert rejected==json.loads((root/'NAME_REJECTIONS.json').read_text())


def test_actual_native_small_label_slice_has_exact_noop_readout():
    p=Path(__file__).resolve().parent/'fixtures/geometry_readout_native_scene0062_64.json'
    d=json.loads(p.read_text());base={k:np.asarray(v,dtype=d['dtypes'][k]) for k,v in d['arrays'].items()}
    assert len(np.unique(base['semantic'][base['semantic']>0]))==7
    out,audit,lineage=apply_geometry_readout(base,base,np.zeros(64,np.int32),np.zeros(64,np.int32))
    assert all(np.array_equal(base[k],out[k]) and base[k].tobytes()==out[k].tobytes() for k in base)
    assert audit['added_points']==0 and len(lineage['point_id'])==0


def test_invalid_geometry_policy_fails_before_output(tmp_path):
    from pose_pipeline.semantic_runtime.fusion import main
    with pytest.raises(ValueError,match='geometry_readout'):
        main(Namespace(arm_root=tmp_path,geometry_readout='typo'))
    assert not (tmp_path/'fused').exists()


def test_gui_refine_compatibility_is_explicit_and_default_refine_still_reaches_runtime(tmp_path,monkeypatch):
    import sys
    from pose_pipeline import live_semantic
    from pose_pipeline.semantic_runtime import pipeline
    class DefaultReached(Exception):pass
    monkeypatch.setattr(live_semantic,'guard_gui_parent',lambda:None)
    monkeypatch.setattr(pipeline,'run',lambda args:(_ for _ in ()).throw(DefaultReached()))
    common=['live_semantic','--manifest',str(tmp_path/'input.json'),'--runtime',str(tmp_path/'runtime.json'),
            '--output',str(tmp_path/'new'),'--refine']
    monkeypatch.setattr(sys,'argv',common+['--geometry-readout','owner-lineage-v1'])
    with pytest.raises(ValueError,match='not yet a validated contract'):live_semantic.main()
    assert not (tmp_path/'new').exists()
    monkeypatch.setattr(sys,'argv',common)
    with pytest.raises(DefaultReached):live_semantic.main()


def test_cli_and_worker_forward_independent_geometry_policy(tmp_path,monkeypatch):
    import argparse,sys
    from pose_pipeline.semantic_runtime.__main__ import add_commands
    from pose_pipeline.semantic_runtime import worker,fusion
    parser=argparse.ArgumentParser();add_commands(parser.add_subparsers(dest='command'))
    args=parser.parse_args(['run-sam3','--manifest','i','--runtime','r','--output','o'])
    assert args.geometry_readout=='off' and args.semantic_confidence_policy=='legacy' and args.semantic_conflict_policy=='consensus'
    runtime=tmp_path/'runtime.json';runtime.write_text('{}');captured=[]
    monkeypatch.setattr(fusion,'main',lambda a:captured.append(a))
    monkeypatch.setattr(sys,'argv',['worker','fusion','--runtime',str(runtime),'--output',str(tmp_path/'new'),
                                    '--geometry-readout','owner-lineage-v1'])
    worker.main();assert captured[0].geometry_readout=='owner-lineage-v1'
    assert captured[0].semantic_confidence_policy=='legacy'


def test_gui_controller_reads_on_contract_and_does_not_hide_sidecar_tampering(tmp_path):
    from types import SimpleNamespace
    from pose_pipeline.live_gui import Controller
    root,base,final,xyz,contract,cloud=prepared(tmp_path)
    export(cloud,root/'map_labels.npz',root/'result.json',root/'export',geometry_contract=contract)
    controller=Controller(SimpleNamespace(replay=None,output=tmp_path/'sessions'))
    controller.session=tmp_path
    controller.state={'status':'completed','pipeline':str(tmp_path),'result':{
        'final_cloud':str(root/'export/map_labeled.ply'),'classes':str(root/'classes.json'),
        'artifacts':str(root/'export/ARTIFACTS.json'),'geometry_readout':'owner-lineage-v1'}}
    assert controller.scene_graph()['nodes'][0]['semantic_id']==1
    p=root/'GEOMETRY_POINT_LINEAGE.npz';p.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='digest mismatch'):controller.scene_graph()

"""Name fixed unknown-object experiments from original multiview cache only."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/src'))
import numpy as np
from pose_pipeline.sam3_object_naming import name_objects, NamingConfig


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    started=time.perf_counter();cfg=NamingConfig()
    plan=json.loads((ROOT/'PLAN_NAMING.json').read_text())
    assert plan['config']==asdict(cfg)
    environment=json.loads((ROOT/'ENV_REUSE.json').read_text())
    assert sha(environment['original_spec_path'])==environment['original_spec_file_sha256']
    sources={str(p):sha(p) for p in [Path(__file__),ROOT/'code/src/pose_pipeline/sam3_object_naming.py']}
    shared_inputs={str(p):sha(p) for p in [ROOT/'PLAN_NAMING.json',ROOT/'ENV_REUSE.json',
        ROOT/'inputs/JOBS.json',Path(environment['original_spec_path'])]}
    rows=[]
    for job in json.loads((ROOT/'inputs/JOBS.json').read_text()):
        t=time.perf_counter();key=job['key'];cache=Path(job['cache_root'])
        result=json.loads((cache/'result.json').read_text())
        selected=job['selected_frame_ids']
        assert result['status']=='completed' and result['selected_frame_ids']==selected
        assert result['processed_frames']==len(selected)
        records=[json.loads(s) for s in (cache/'frames.jsonl').read_text().splitlines() if s]
        assert [r['frame_id'] for r in records]==selected and len(set(selected))==len(selected)
        paths=[cache/'frames'/f'{fid:06}.npz' for fid in selected]
        assert set(paths)==set((cache/'frames').glob('*.npz'))
        inputs={**shared_inputs,**{str(p):sha(p) for p in [cache/'result.json',cache/'frames.jsonl',*paths]}}
        frames=[]
        for fid,path in zip(selected,paths):
            with np.load(path,allow_pickle=False) as z:
                frame={k:z[s].copy() for k,s in {'point_ids':'visible_map_ids',
                    'semantic':'projected_semantic','confidence':'projected_confidence','interior':'interior'}.items()}
            frame['frame_id']=fid;frames.append(frame)
        preparation=time.perf_counter()-t
        for stage in plan['base_arms']:
            arm_started=time.perf_counter();base=ROOT/stage/key
            baseline_paths=[base/f for f in ['result.json','objects.json','map_labels.npz','classes.json']]
            arm_inputs={**inputs,**{str(p):sha(p) for p in baseline_paths}}
            before=json.loads((base/'result.json').read_text())
            assert before['status']=='completed' and before['geometry_xyz_sha256']==job['geometry_xyz_sha256']
            assert before['selected_frame_ids']==selected and before['processed_frames']==len(selected)
            assert before['semantic_unchanged'] and before['confidence_unchanged']
            original=json.loads((base/'objects.json').read_text())
            original={obj['instance_id']:obj for obj in original}
            dictionary=json.loads((base/'classes.json').read_text())
            with np.load(base/'map_labels.npz',allow_pickle=False) as z:
                instance=z['instance'].copy()
            assert len(instance)==job['expected_points']
            inference_started=time.perf_counter()
            named,audit=name_objects(instance,frames,cfg)
            elapsed=time.perf_counter()-inference_started
            assert set(original)=={obj['instance_id'] for obj in named}
            output=[]
            for obj in named:
                old=original[obj['instance_id']]
                assert old['point_count']==obj['point_count']
                label=str(obj['semantic_id'])
                assert label in dictionary, 'unsupported cache class; cannot invent label name'
                output.append({**old,**obj,'semantic_name':dictionary[label],
                    'source_metadata_semantic_id':old.get('semantic_id'),
                    'source_metadata_semantic_id_policy':old.get('semantic_id_policy'),
                    'semantic_id_policy':'fixed multiview object naming; point semantic labels unchanged',
                    'semantic_naming_executed':True,
                    'naming_method':'explicit cached multiview frame voting',
                    'point_semantic_labels_unchanged':True,
                    'point_semantic_histogram':old['semantic_histogram']})
            assert all(sha(p)==h for p,h in {**arm_inputs,**sources}.items()),'frozen input/source changed'
            write_new(base/'objects_named.json',output)
            report={'status':'completed','key':key,'base_arm':stage,'stage':'explicit_multiview_object_naming',
                'utc':datetime.now(timezone.utc).isoformat(),'protocol_extension':True,
                'blind_confirmatory_experiment':False,'config':asdict(cfg),
                'audit':audit,'base_result_sha256':arm_inputs[str(base/'result.json')],
                'input_sha256':arm_inputs,'source_files_sha256':sources,
                'output_file':'objects_named.json','output_sha256':sha(base/'objects_named.json'),
                'input_hashes_verified_after_run':True,
                'point_labels_and_instance_and_geometry_preserved':True,
                'base_objects_json_unchanged':True,'semantic_map_changed':False,
                'new_model_inference':False,'sam3_inference_executed':False,
                'sga_inference_executed':False,'clip_inference_executed':False,'llm_inference_executed':False,
                'gt_consumed':False,'new_name_strings_invented':False,
                'coarse_classes_remapped_to_fine':False,
                'selected_frame_ids':selected,'processed_frames':len(frames),
                'complete_selected_frames':True,'complete_full_sequence':False,
                'scope':job.get('scope','frozen selected cached frames'),
                'naming_inference_seconds':elapsed,'cache_preparation_seconds':preparation,
                'arm_seconds':time.perf_counter()-arm_started,
                'quality_accepted':False,'environment':{'python':sys.executable,
                    'python_version':platform.python_version(),'numpy':np.__version__,
                    'reuse_receipt':str(ROOT/'ENV_REUSE.json')}}
            write_new(base/'naming_result.json',report)
            row={'key':key,'base_arm':stage,'objects':audit['objects'],'named_objects':audit['named_objects'],
                'unknown_objects':audit['unknown_objects'],'named_points':sum(o['point_count'] for o in named if o['semantic_id']>0),
                'named_class_object_counts':{label:sum(str(o['semantic_id'])==label for o in named) for label in dictionary
                    if any(str(o['semantic_id'])==label for o in named)},'naming_inference_seconds':elapsed}
            rows.append(row);print(json.dumps(row),flush=True)
    write_new(ROOT/'NAMING_SUMMARY.json',{'status':'completed','runs':len(rows),'rows':rows,
        'seconds':time.perf_counter()-started,'gt_consumed':False,'quality_accepted':False})


if __name__=='__main__':main()

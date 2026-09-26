"""Post-hoc attribution and downstream behavior audit; no prediction edits."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_context_review_trial import config,base,read,write,sha
from pose_pipeline.scene_graph import build_graph,query_graph,_label


def main():
    assert read(ROOT/'evaluation/COMPLETE.json')['status']=='completed'
    base.verify(read(ROOT/'INPUT_LOCK.json'));base.verify(read(ROOT/'PREDICTIONS_LOCK.json'))
    rows=[];fallback=[];models=set();completion=[]
    for scene in config()['scenes']:
        folder=ROOT/'runs/repeat-2'/scene;b=folder/'B';d=folder/'D'
        labels=np.load(b/'fused/map_labels.npz');xyz=np.load(b/'fused/target.npz')['xyz']
        classes=read(b/'fused/classes.json');bn=read(b/'fused/instance_names.json')
        dn=read(d/'fused/instance_names.json');en=read(d/'fused/instance_names_E.json')
        maps={'B':bn,'D':dn,'E':en};graphs={}
        for arm,names in maps.items():
            graphs[arm]=build_graph(xyz,labels['semantic'],labels['instance'],classes,confidence=labels['confidence'],names=names)
        assert graphs['B']['edges']==graphs['D']['edges']==graphs['E']['edges']
        vocabulary={_label(v) for v in classes.values() if v!='unknown'}
        vocabulary.update(_label(n['vlm_name']) for ns in maps.values() for n in ns if n['vlm_name']!='unknown')
        bq={v:query_graph(graphs['B'],label=v)['instance_ids'] for v in sorted(vocabulary)}
        for arm in ('D','E'):
            plain=[];spatial=[];spatial_count=0
            for v in sorted(vocabulary):
                predicted=query_graph(graphs[arm],label=v)['instance_ids']
                if predicted!=bq[v]:plain.append({'label':v,'B':bq[v],'candidate':predicted})
            for node in graphs['B']['nodes']:
                oid=node['instance_id']
                for v in ('table','door','chair','cabinet','mirror','television','trash can','gravestone'):
                    for options in ({'nearest_to':oid},{'relation':'near','reference_id':oid}):
                        x=query_graph(graphs['B'],label=v,**options)['instance_ids']
                        y=query_graph(graphs[arm],label=v,**options)['instance_ids'];spatial_count+=1
                        if x!=y:spatial.append({'label':v,**options,'B':x,'candidate':y})
            rows.append({'scene':scene,'arm':arm,'label_queries_compared':len(vocabulary),'label_query_changes':plain,
                         'spatial_queries_compared':spatial_count,'spatial_query_changes':spatial,
                         'comparison':'returned instance IDs, not evidence metadata; fixed measured spatial graph'})
        # Explanatory control AFTER main evaluation: display existing SAM3
        # class when VLM metadata is unknown, without invoking a model.
        frames=defaultdict(set)
        for c in read(b/'OBJECT_STORE.json')['crops']:
            if c['association']['eligible']:frames[c['association']['object_id']].add(c['frame_id'])
        effective={n['instance_id']:(classes[str(n['semantic_id'])] if n['vlm_name']=='unknown' and len(frames[n['instance_id']])>=2
                                    else n['vlm_name']) for n in bn}
        targets={x['object_id']:_label(x['canonical_GT_name']) for x in read(ROOT/'evaluation'/(scene+'-GT-targets.json')) if x['eligible']}
        fallback.append({'scene':scene,'GT_eligible':len(targets),
                         'GT_correct':sum(_label(effective[i])==target for i,target in targets.items()),
                         'extra_VLM_calls':0,'scope':'post-hoc SAM3 display fallback attribution control; not a new VLM prediction or promoted policy'})
        for repeat in (1,2):
            dd=ROOT/'runs'/f'repeat-{repeat}'/scene/'D'
            data={a:read(dd/'fused'/file) for a,file in [('D','instance_names.json'),('E','instance_names_E.json')]}
            complete={'status':'completed','base_backfill_receipt':'NAME_COMPLETE.json describes B backfill before review',
                      'final_named_objects':{a:sum(n['vlm_name']!='unknown' for n in ns) for a,ns in data.items()},
                      'final_name_sha256':{a:sha(dd/'fused'/file) for a,file in [('D','instance_names.json'),('E','instance_names_E.json')]},
                      'role':'derived completion audit; no prediction modifications'}
            write(dd/'fused/CONTEXT_COMPLETE.json',complete);completion.append(complete)
            for arm in ('B','D'):
                model=read(dd.parent/arm/'semantic/vlm/MODEL.json')
                models.add((model['id'],model['revision'],model['receipt_sha256'],model['torch'],model['transformers']))
    assert len(models)==1
    write(ROOT/'evaluation/DOWNSTREAM_QUERY_AUDIT.json',rows)
    write(ROOT/'evaluation/POSTHOC_SAM3_FALLBACK.json',fallback)
    summary={a:{'label_queries_compared':sum(x['label_queries_compared'] for x in rows if x['arm']==a),
                'label_queries_changed':sum(len(x['label_query_changes']) for x in rows if x['arm']==a),
                'spatial_queries_compared':sum(x['spatial_queries_compared'] for x in rows if x['arm']==a),
                'spatial_queries_changed':sum(len(x['spatial_query_changes']) for x in rows if x['arm']==a)} for a in ('D','E')}
    base.verify(read(ROOT/'INPUT_LOCK.json'));base.verify(read(ROOT/'PREDICTIONS_LOCK.json'));base.verify(read(ROOT/'evaluation/GT_LOCK.json'))
    write(ROOT/'FINAL_AUDIT.json',{'status':'passed','prediction_commit':'c9308bbce03a8d6fead458064a7cf7dd1ac3717c',
        'input_files':len(read(ROOT/'INPUT_LOCK.json')),'sealed_prediction_files':len(read(ROOT/'PREDICTIONS_LOCK.json')),
        'all_locked_inputs_predictions_GT_unchanged':True,'model_identities':[list(x) for x in models],
        'queries':summary,'posthoc_SAM3_fallback_GT_correct':sum(x['GT_correct'] for x in fallback),
        'posthoc_SAM3_fallback_GT_eligible':sum(x['GT_eligible'] for x in fallback),'production_modified':False})
    print(json.dumps({'queries':summary,'posthoc_control':fallback},indent=2))


if __name__=='__main__':main()

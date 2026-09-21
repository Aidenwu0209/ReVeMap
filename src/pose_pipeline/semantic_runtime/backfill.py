"""Associate observed crop names with measured final instances; keep base classes."""
from pathlib import Path
import argparse,time,sys
from collections import Counter,defaultdict
from .common import read, write, event

def main(args):
    import numpy as np
    arm=Path(args.arm_root);start=time.monotonic();event(arm,'name_backfill_start')
    labels=np.load(arm/'fused/map_labels.npz');inst=labels['instance'];sem=labels['semantic'];evidence=defaultdict(list);rejected=[]
    for p in sorted((arm/'semantic/vlm').glob('response_*.json')):
        response=read(p)
        for crop in response['crops']:
            f=np.load(arm/'fused'/f'projection_{crop["frame_id"]:06}.npz');points=f['point_ids'][f['mask_ids']==crop['mask_id']]
            counts=Counter(int(x) for x in inst[points] if x>0)
            if not counts:continue
            oid,count=counts.most_common(1)[0];purity=count/max(1,len(points))
            rec={**crop,'candidate_instance_id':oid,'projected_instance_share':purity,'visible_points':len(points),'model_completed_at':response['completed_at']}
            if count>=30 and purity>=.65 and crop['label']!='unknown':evidence[oid].append(rec)
            else:rejected.append(rec)
    objects=[]
    for oid in sorted(int(x) for x in np.unique(inst) if x>0):
        ev=evidence[oid];bylabel=defaultdict(set)
        for x in ev:bylabel[x['label']].add(x['frame_id'])
        ranked=sorted(bylabel,key=lambda k:(-len(bylabel[k]),k));name='unknown'
        if ranked and len(bylabel[ranked[0]])>=2 and (len(ranked)==1 or len(bylabel[ranked[0]])>len(bylabel[ranked[1]])):name=ranked[0]
        classes,n=np.unique(sem[inst==oid],return_counts=True)
        objects.append({'instance_id':oid,'semantic_id':int(classes[n.argmax()]),'point_count':int(np.sum(inst==oid)),
            'vlm_name':name,'support_frames':sorted(bylabel.get(name,[])),'evidence':ev,
            'assignment_scope':'multiview naming metadata; original semantic_id retained, no new SAM3 text grounding or geometry change'})
    write(arm/'fused/instance_names.json',objects);write(arm/'fused/NAME_REJECTIONS.json',rejected)
    write(arm/'fused/NAME_COMPLETE.json',{'seconds':time.monotonic()-start,'completed_at':time.monotonic(),'named_instances':sum(x['vlm_name']!='unknown' for x in objects),'total_instances':len(objects)})
    event(arm,'name_backfill_complete',named=sum(x['vlm_name']!='unknown' for x in objects))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm-root',required=True);main(p.parse_args())

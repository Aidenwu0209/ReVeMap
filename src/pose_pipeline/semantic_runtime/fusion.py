"""Project fresh masks onto this arm's freshly reconstructed map, then associate."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import sys,time,argparse,hashlib,traceback
from pathlib import Path
from .common import CONFIG_ROOT, read, write, event, sha

def main(args):
    import numpy as np
    from plyfile import PlyData
    from reconstruction.rgbd_refusion import _read_rgbd
    from pose_pipeline.contracts import load_manifest,load_trajectory,bind_manifest_trajectory
    from pose_pipeline.sam3_fusion import MapVotes,GeometricInstances,visible_map_pixels
    from pose_pipeline.sam3_refine import interior
    from pose_pipeline.instance_association import select_tracks,compatible,measure_candidates,greedy_pairs,object_consensus
    from pose_pipeline.sam3_guided import recover_instances
    from pose_pipeline.sam3_export import export
    from .multiview import fuse_instances
    arm=Path(args.arm_root);out=arm/'fused';out.mkdir(exist_ok=False)
    started=time.monotonic();event(arm,'projection_fusion_start')
    geom=read(arm/'mapping/mapping_result.json');m=load_manifest(Path(geom['manifest']))
    poses,_=load_trajectory(Path(geom['trajectory']));bound={f.frame_id:(f,p) for f,p in bind_manifest_trajectory(m,poses)}
    cloud=Path(geom['final_cloud']);verts=PlyData.read(cloud)['vertex'].data
    xyz=np.stack([verts[c] for c in ['x','y','z']],axis=1);n=len(xyz)
    votes=MapVotes(n,35);high=np.zeros(n,np.float32);streams=[GeometricInstances(),GeometricInstances()];frames=[];projections=[]
    for ordinal,row in enumerate(read(arm/'semantic/FRAMES.json')):
        fid=row['frame_id'];f,p=bound[fid];_,depth,K=_read_rgbd(f)
        if sha(f.color_path)!=row['color_sha256'] or sha(f.depth_path)!=row['depth_sha256']:raise ValueError('RGB-D changed since SAM3 inference')
        if sha(arm/'semantic/frames'/f'{fid:06}.npz')!=row['mask_sha256']:raise ValueError('SAM3 cache digest mismatch')
        ids,v,u=visible_map_pixels(xyz,p.t_world_camera,K,depth.astype(float)/m.depth_scale)
        with np.load(arm/'semantic/frames'/f'{fid:06}.npz') as z:
            frame={'frame_id':fid,'point_ids':ids,'mask_ids':z['local_instance'][v,u],'semantic':z['semantic'][v,u],'confidence':z['confidence'][v,u],'interior':interior(z['semantic'])[v,u]}
        safe=frame['interior'];votes.add(fid,ids,frame['semantic'],frame['confidence'])
        high[ids[safe]]=np.maximum(high[ids[safe]],frame['confidence'][safe])
        streams[ordinal%2].add(fid,ids[safe],frame['mask_ids'][safe],frame['semantic'][safe]);frames.append(frame)
        np.savez_compressed(out/f'projection_{fid:06}.npz',point_ids=ids,mask_ids=frame['mask_ids'],semantic=frame['semantic'],row=v,col=u)
        projections.append({'frame_id':fid,'depth_consistent_points':len(ids),'projected_at':time.monotonic()})
    sem,conf,_=votes.finalize();single=(sem==0)&(votes.counts.sum(1)==1)&(high>=.9)
    sem[single]=votes.scores.argmax(1)[single];conf[single]=high[single]
    trackrecords=[]
    for tr in streams:
        trackrecords.append([{'track_id':i+1,'category':t['category'],'frames':sorted(t['frames']),'points':sorted(t['points']),'mean_point_score':float(high[np.array(sorted(t['points']),np.int64)].mean())} for i,t in enumerate(tr.tracks)])
    chosen=[select_tracks(t) for t in trackrecords]
    if min(map(len,chosen))<2:pairs=[]
    else:
        candidates=[(a['track_id'],b['track_id'],0.) for a in chosen[0] for b in chosen[1] if compatible(a['category'],b['category'])]
        pairs=greedy_pairs(measure_candidates(candidates,*chosen,xyz),'geometry')
    c,_,cm=object_consensus(trackrecords,pairs,{'semantic':sem,'confidence':conf},xyz)
    cfg={'min_mask_points':30,'min_output_points':50,'min_point_views':1,'min_group_frames':2,'object_score_mode':'max_point'}
    current,audit=fuse_instances(n,frames,c['semantic'].copy(),config=cfg)
    blocked=[(int(x['frame_id']),int(x['mask_id'])) for x in audit['filtered_masks']]
    inst,ga,provenance=recover_instances(n,frames,c['semantic'].copy(),c['instance'].copy(),current,blocked_masks=blocked)
    assert np.array_equal(inst[current>0],current[current>0]);assert np.all(provenance['distinct_support_frames']>=2)
    labels={**c,'instance':inst};np.savez_compressed(out/'map_labels.npz',**labels)
    np.savez_compressed(out/'target.npz',xyz=xyz)
    write(out/'PROJECTIONS.json',projections);write(out/'CONSENSUS.json',{'pairs':pairs,'metrics':cm});write(out/'MULTIVIEW.json',audit);write(out/'GUIDED.json',ga)
    result={'status':'completed','map_points':n,'semantic_coverage':float(np.mean(c['semantic']>0)),'instance_coverage':float(np.mean(inst>0)),
        'instance_count':len(np.unique(inst[inst>0])),'selected_frames':len(frames),'geometry_xyz_sha256':hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest(),
        'sga_inference_executed':False,'sgf_prior_used':False,'complete_full_sequence':True,'raw_window_complete':True,'GT_used':False,'geometry_modified':False,
        'pipeline_scope':'fresh raw map + fixed SAM3 concepts + measured geometry association + multiview consensus + guided recovery; no frozen-map birth/completion or SGF subtype prior',
        'seconds':time.monotonic()-started,'completed_at':time.monotonic()}
    write(out/'classes.json', {'0':'unknown', **{str(c['id']):c['name'] for c in read(CONFIG_ROOT/'sam3_indoor_v1.json')['classes']}})
    write(out/'result.json',result);export(cloud,out/'map_labels.npz',out/'result.json',out/'export')
    write(out/'EXPORT_COMPLETE.json',{'completed_at':time.monotonic(),'seconds_including_export':time.monotonic()-started})
    event(arm,'projection_fusion_complete',points=n)
    print('FUSED',result,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm-root',required=True);a=p.parse_args()
    try:main(a)
    except BaseException:write(Path(a.arm_root)/'FUSION_FAILURE.json',{'error':traceback.format_exc()});raise

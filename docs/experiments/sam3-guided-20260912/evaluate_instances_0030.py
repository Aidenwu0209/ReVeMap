"""Fixed-domain instance audit, explicitly not official ScanNet instance AP."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parent;C=R.parent
p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
G=Path('/Users/wu/Desktop/wu/Xia/jojo_updated_scene0030_run_20260902/source/Jojo_current_panoptic_scene_graph_aligment_on_full_scene0030_00/full_scene0030_00_dataset/gt_scannet_data_scene0030_00')
read=lambda p:json.loads(p.read_text());xyz=lambda v:np.stack([v[k] for k in ['x','y','z']],axis=1)
gtfile=G/'scene0030_00_vh_clean_2.ply';segfile=G/'scene0030_00_vh_clean_2.0.010000.segs.json';groupfile=G/'scene0030_00.aggregation.json'
gx=xyz(PlyData.read(gtfile)['vertex'].data);segments=np.array(read(segfile)['segIndices'])
classes={v:int(k) for k,v in read(C/'sgf_sga_all_scannet_orbbec_20260910_v1/scenes/scannet/scene0030_00/map/classes.json').items()}
y=np.zeros(len(gx),int);g=np.zeros(len(gx),int);gt_names={}
for item in read(groupfile)['segGroups']:
 mask=np.isin(segments,item['segments']);label=classes.get(item['label'],0);gid=item['objectId']+1
 y[mask]=label;g[mask]=gid;gt_names[gid]=item['label']
thing=(y>0)&~np.isin(y,[10,19])
eligible_ids=[i for i in np.unique(g[thing]) if np.sum(thing&(g==i))>=50]
eligible=thing&np.isin(g,eligible_ids)
v=PlyData.read(args.map)['vertex'].data
alignment=C/'scannet0030_semantic_comparison_20260909_v1/developnew_full/evaluation/diagnostic.json'
T=np.array(read(alignment)['T_dataset_estimated_world']);x=xyz(v)@T[:3,:3].T+T[:3,3];dist,ix=cKDTree(x).query(gx,workers=1)
pred=np.where(dist<=.05,v['instance_id'][ix],0);sem=v['semantic_id'][ix]
known=eligible&(pred>0);pred_ids=np.unique(pred[known]);truth_ids=np.array(eligible_ids)
intersections=np.zeros((len(truth_ids),len(pred_ids)),np.int64)
if len(pred_ids):
 gi=np.searchsorted(truth_ids,g[known]);pi=np.searchsorted(pred_ids,pred[known])
 intersections=np.bincount(gi*len(pred_ids)+pi,minlength=len(truth_ids)*len(pred_ids)).reshape(intersections.shape)
gsize=np.array([np.sum(eligible&(g==i)) for i in truth_ids]);psize=intersections.sum(axis=0)
unions=gsize[:,None]+psize[None,:]-intersections
iou=np.divide(intersections,unions,out=np.zeros_like(intersections,dtype=float),where=unions>0)
if len(pred_ids):
 class_count=int(sem.max())+1
 pred_classes=np.bincount(pi*class_count+sem[known],minlength=len(pred_ids)*class_count).reshape(len(pred_ids),class_count).argmax(axis=1)
 for i,gid in enumerate(truth_ids):iou[i,pred_classes!=classes[gt_names[gid]]]=0
assigned=list(zip(*linear_sum_assignment(-iou))) if iou.size else []
valid_predictions=psize>=50
mixed=0
for j in np.flatnonzero(valid_predictions):
 fractions=np.sort(intersections[:,j]/psize[j])
 if len(fractions)>1 and fractions[-2]>=.1:mixed+=1
per_object=[]
for i,gid in enumerate(truth_ids):
 per_object.append({'gt_object_id':int(gid-1),'class':gt_names[gid],'gt_points':int(gsize[i]),
   'fragments_covering_at_least_5pct':int(np.sum(intersections[i]>=.05*gsize[i])),
   'largest_fragment_recall':float(intersections[i].max()/gsize[i]) if len(pred_ids) else 0})
out={'scope':'fixed eligible GT objects >=50 vertices, excluding wall/floor; old evaluation-only alignment, 5cm gate; not official instance AP',
 'eligible_GT_vertices':int(eligible.sum()),'eligible_GT_objects':len(truth_ids),'instance_known_coverage':float(known.sum()/eligible.sum()),
 'predicted_instances_touching_GT':len(pred_ids),'predicted_instances_ge50_GT_points':int(valid_predictions.sum()),
 'weighted_instance_purity':float(intersections.max(axis=0).sum()/max(1,intersections.sum())) if intersections.size else 0,
 'mixed_instances_secondary_GT_fraction_ge10pct':mixed,
 'mean_fragments_per_GT_object':float(np.mean([o['fragments_covering_at_least_5pct'] for o in per_object])),
 'mean_largest_fragment_recall':float(np.mean([o['largest_fragment_recall'] for o in per_object])),
 'one_to_one_same_class_recall_IoU25':sum(iou[i,j]>=.25 for i,j in assigned)/len(truth_ids),
 'one_to_one_same_class_recall_IoU50':sum(iou[i,j]>=.5 for i,j in assigned)/len(truth_ids),
 'per_object':per_object,'map':str(args.map.resolve()),'map_sha256':hashlib.sha256(args.map.read_bytes()).hexdigest(),
 'GT_in_inference':False}
args.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['per_object','map','map_sha256']},indent=2))

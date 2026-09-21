"""Read GT strictly after inference; fixed evaluator and full vertex denominator."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent;C=R.parent
p=argparse.ArgumentParser();p.add_argument('--sam-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
GT=Path('/Users/wu/Desktop/wu/Xia/jojo_updated_scene0030_run_20260902/source/Jojo_current_panoptic_scene_graph_aligment_on_full_scene0030_00/full_scene0030_00_dataset/gt_scannet_data_scene0030_00')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();xyz=lambda v:np.stack([v[k] for k in ['x','y','z']],axis=1)
classes={int(k):v for k,v in read(C/'sgf_sga_all_scannet_orbbec_20260910_v1/scenes/scannet/scene0030_00/map/classes.json').items()};names={v:k for k,v in classes.items() if k>0}
paths=[GT/'scene0030_00_vh_clean_2.ply',GT/'scene0030_00_vh_clean_2.0.010000.segs.json',GT/'scene0030_00.aggregation.json',C/'scannet0030_semantic_comparison_20260909_v1/developnew_full/evaluation/diagnostic.json']
inputs={str(p):sha(p) for p in paths};gx=xyz(PlyData.read(paths[0])['vertex'].data);segments=np.array(read(paths[1])['segIndices']);y=np.zeros(len(gx),int)
for ob in read(paths[2])['segGroups']:y[np.isin(segments,ob['segments'])]=names.get(ob['label'],0)
eligible=y>0;assert eligible.sum()==185436
T=np.array(read(paths[3])['T_dataset_estimated_world']);methods={}
for method,mp in {'sgf':C/'sgf_sga_all_scannet_orbbec_20260910_v1/scenes/scannet/scene0030_00/map/map_labeled.ply','sam3':args.sam_map}.items():
 inputs[str(mp)]=sha(mp);v=PlyData.read(mp)['vertex'].data;x=xyz(v)@T[:3,:3].T+T[:3,3];d,ix=cKDTree(x).query(gx,workers=1);geometry=eligible&(d<=.05);pred=v['semantic_id'][ix];known=geometry&(pred>0);correct=known&(pred==y);common=known&(pred<=20)
 per_class={}
 for k in sorted(names.values()):
  if not np.any(y==k):continue
  truth=eligible&(y==k);guess=geometry&(pred==k);tp=int(np.sum(truth&guess));fp=int(np.sum(eligible&~truth&guess));fn=int(np.sum(truth&~guess));per_class[classes[k]]={'GT_points':int(truth.sum()),'correct':tp,'false_positive':fp,'recall':tp/max(1,int(truth.sum())),'iou_fixed_eligible':tp/max(1,tp+fp+fn)}
 methods[method]={'map_points':len(v),'eligible_GT':int(eligible.sum()),'geometry_coverage':float(geometry.sum()/eligible.sum()),'known_coverage_all_classes':float(known.sum()/eligible.sum()),'known_coverage_common20':float(common.sum()/eligible.sum()),'correct_coverage_common20':float(correct.sum()/eligible.sum()),'conditional_accuracy_all_labels':float(correct.sum()/max(1,known.sum())),'conditional_accuracy_common20_labels':float(correct.sum()/max(1,common.sum())),'novel_label_GT_points':int(np.sum(known&(pred>20))),'per_class':per_class}
assert all(sha(Path(p))==h for p,h in inputs.items())
out={'methods':methods,'input_sha256':inputs,'scope':'same 185436 eligible GT vertices; geometry5cm; fixed earlier evaluation-only alignment; not official ScanNet mIoU/AP; appended SAM classes reported separately','ground_truth_in_inference':False,'adopted':False}
args.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a!='per_class'} for k,v in methods.items()},indent=2))

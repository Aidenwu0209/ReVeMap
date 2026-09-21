"""Post-run label diagnostic, with explicit first-frame GT gauge alignment.

This is a conditional nearest-surface diagnostic, NOT benchmark mIoU/AP.
GT never enters run-semantic; inputs are verified unchanged after evaluation.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from pose_pipeline.contracts import load_manifest, load_trajectory, sha256_file
def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert,
                                    allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('result','manifest','trajectory','reference-dir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    receipt = json.loads((a.result/'result.json').read_text())
    if receipt['status']!='completed':
        raise ValueError('prediction must be completed before GT evaluation')
    inputs = [a.result/'map_labeled.ply',a.result/'result.json',a.manifest,a.trajectory]
    before = {str(x):sha256_file(x) for x in inputs}
    manifest = load_manifest(a.manifest)
    poses,_ = load_trajectory(a.trajectory)
    reference = a.reference_dir
    gt_files = []
    if manifest.dataset=='3rscan':
        mesh = reference/'labels.instances.annotated.v2.ply'
        semseg = reference/'semseg.v2.json'
        gt_files += [mesh,semseg]
        gt = PlyData.read(mesh)['vertex'].data
        object_ids = np.asarray(gt['objectId'])
        names = {int(o['objectId']):o['label'] for o in json.loads(semseg.read_text())['segGroups']}
    elif manifest.dataset=='scannet':
        scene = manifest.sequence_id
        mesh = reference/(scene+'_vh_clean_2.ply')
        aggregate = reference/(scene+'.aggregation.json')
        segments = reference/(scene+'_vh_clean_2.0.010000.segs.json')
        gt_files += [mesh,aggregate,segments]
        gt = PlyData.read(mesh)['vertex'].data
        seg = np.array(json.loads(segments.read_text())['segIndices'])
        if len(seg)!=len(gt):
            raise ValueError('ScanNet mesh/segment vertex mismatch')
        object_ids = np.zeros(len(gt),np.int64)
        names = {}
        for obj in json.loads(aggregate.read_text())['segGroups']:
            oid = int(obj['objectId'])+1
            names[oid] = obj['label']
            object_ids[np.isin(seg,obj['segments'])] = oid
    else:
        raise ValueError('No verified Orbbec GT supported')
    for pose in poses:
        path = (manifest.root/'pose'/f'{pose.frame_id}.txt' if manifest.dataset=='scannet'
                else manifest.root/f'frame-{pose.frame_id:06d}.pose.txt')
        matrix = np.loadtxt(path)
        if not np.isfinite(matrix).all():
            continue
        if manifest.dataset=='3rscan' and manifest.frames[0].rotate_ccw:
            rot = np.eye(4);rot[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
            matrix = matrix@rot
        alignment = matrix@np.linalg.inv(pose.t_world_camera)
        gt_files.append(path)
        break
    else:
        raise ValueError('No finite GT alignment pose')
    pred = PlyData.read(a.result/'map_labeled.ply')['vertex'].data
    xyz = np.column_stack([pred[k] for k in ('x','y','z')])
    world = xyz@alignment[:3,:3].T + alignment[:3,3]
    gtxyz = np.column_stack([gt[k] for k in ('x','y','z')])
    distance,nearest = cKDTree(gtxyz).query(world,workers=1)
    class_dict = json.loads((a.result/'classes.json').read_text())
    vocab = {name:int(i) for i,name in class_dict.items() if int(i)>0}
    # Exact names only. Out-of-vocabulary labels are explicitly excluded.
    reference_class = np.array([vocab.get(names.get(int(i),''),0) for i in object_ids[nearest]])
    supported = (distance<=.05)&(reference_class>0)
    labeled = supported&(pred['semantic_id']>0)
    per_class = {}
    for name, cid in vocab.items():
        target = supported&(reference_class==cid)
        if target.any():
            per_class[name] = {'reference_supported_points':int(target.sum()),
                'correct':int(np.sum(target&(pred['semantic_id']==cid))),
                'unknown':int(np.sum(target&(pred['semantic_id']==0)))}
    purity_sum, assigned, dominant = 0,0,Counter()
    for iid in np.unique(pred['instance_id']):
        if iid==0:continue
        mask = (pred['instance_id']==iid)&(distance<=.05)&(object_ids[nearest]>0)
        if mask.sum()<20:continue
        counts = Counter(object_ids[nearest[mask]].tolist())
        oid, count = counts.most_common(1)[0]
        purity_sum += count; assigned += int(mask.sum()); dominant[int(oid)] += 1
    report = {'status':'completed','gt_role':'post_prediction_evaluation_only',
        'metric_scope':'predicted vertices within 5cm of GT; exact shared class names; not benchmark mIoU/AP',
        'alignment':'first finite pose, no ICP, no scale fitting', 'alignment_frame':pose.frame_id,
        'T_dataset_estimated_world':alignment, 'prediction_points':len(pred),
        'near_reference_points':int((distance<=.05).sum()),
        'supported_class_points':int(supported.sum()), 'labeled_supported_points':int(labeled.sum()),
        'semantic_accuracy_with_unknown_as_error':float(np.mean(pred['semantic_id'][supported]==reference_class[supported])) if supported.any() else None,
        'semantic_accuracy_labeled_only':float(np.mean(pred['semantic_id'][labeled]==reference_class[labeled])) if labeled.any() else None,
        'weighted_instance_purity_min20points':purity_sum/assigned if assigned else None,
        'predicted_fragments_per_dominant_gt_object':dict(dominant), 'per_class':per_class,
        'input_sha256':before,'reference_sha256':{str(x):sha256_file(x) for x in gt_files}}
    if before!={str(x):sha256_file(x) for x in inputs}:
        raise RuntimeError('prediction changed during evaluation')
    report['prediction_unchanged'] = True
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'diagnostic.json',report)


if __name__=='__main__':
    main()

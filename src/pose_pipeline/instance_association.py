"""Measured object association extracted from developnew without SGF/SGAligner.

These helpers keep the original geometric thresholds and object consensus.
ICP is validation evidence only; it does not change the trajectory.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

def measured_association(a, b):
    """Shared-frame overlap plus an ICP measurement; never apply its pose."""
    import open3d as o3d
    da = cKDTree(b).query(a, workers=1)[0]
    db = cKDTree(a).query(b, workers=1)[0]
    coverage = min(float(np.mean(da < .05)), float(np.mean(db < .05)))
    if coverage < .3:
        return {"accepted": False, "coverage_5cm": coverage, "reason": "overlap"}
    def pc(x):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(x)
        return p.voxel_down_sample(.02)
    fit = o3d.pipelines.registration.registration_icp(
        pc(a), pc(b), .05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))
    transform = fit.transformation
    angle = float(np.degrees(np.arccos(np.clip((np.trace(transform[:3,:3])-1)/2, -1, 1))))
    translation = float(np.linalg.norm(transform[:3, 3]))
    accepted = bool(fit.fitness >= .3 and fit.inlier_rmse <= .03
                    and translation <= .10 and angle <= 5)
    return {"accepted": accepted, "coverage_5cm": coverage,
            "fitness": fit.fitness, "rmse_m": fit.inlier_rmse,
            "translation_m": translation, "rotation_deg": angle,
            "T_ref_src_measured_only": transform, "applied_to_trajectory": False}


def compatible(a, b):
    return a == b or (a == 33 and b in (8,17)) or (b == 33 and a in (8,17)) or (a == 34 and b in (7,14)) or (b == 34 and a in (7,14))


def select_tracks(tracks, limit=128):
    # Context chosen without GT. Small/one-view objects remain in exports, but
    # cannot displace all stable context in a bounded learned descriptor pass.
    eligible=[t for t in tracks if len(t['points']) >= 50]
    return sorted(eligible,key=lambda t:(-len(t['frames']),-len(t['points']),t['track_id']))[:limit]


def measure_candidates(candidates, tracks_a, tracks_b, xyz):
    ta={t['track_id']:t for t in tracks_a};tb={t['track_id']:t for t in tracks_b}
    records=[]
    for a,b,score in candidates:
        x,y=ta[a],tb[b]
        r={'source_track':a,'target_track':b,'rank_score':float(score),'accepted_geometry':False}
        if not compatible(x['category'],y['category']):r['reason']='incompatible_category'
        else:
            p=xyz[x['points']];q=xyz[y['points']]
            if np.any(p.min(axis=0)>q.max(axis=0)+.05) or np.any(q.min(axis=0)>p.max(axis=0)+.05):r['reason']='disjoint_bounds'
            else:
                evidence=measured_association(p,q)
                r.update(evidence);r['accepted_geometry']=bool(evidence['accepted'])
        records.append(r)
    return records


def greedy_pairs(records, method):
    valid=[r for r in records if r['accepted_geometry']]
    if method=='geometry':valid.sort(key=lambda r:(-r['coverage_5cm'],r['rmse_m'],r['source_track'],r['target_track']))
    else:valid.sort(key=lambda r:(r['rank_score'],r['source_track'],r['target_track']))
    used_a=set();used_b=set();accepted=[]
    for r in valid:
        a,b=r['source_track'],r['target_track']
        if a in used_a or b in used_b:continue
        used_a.add(a);used_b.add(b);accepted.append((a,b))
    return accepted


def object_consensus(streams, pairs, base, xyz):
    """Propagate within measured mask unions; preserve B's known fine labels."""
    keys=[(s,t['track_id']) for s,rows in enumerate(streams) for t in rows]
    tracks={(s,t['track_id']):t for s,rows in enumerate(streams) for t in rows}
    parent={k:k for k in keys}
    def root(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    for a,b in pairs:
        ka,kb=(0,a),(1,b)
        if not compatible(tracks[ka]['category'],tracks[kb]['category']):raise ValueError('incompatible merge')
        parent[root(kb)]=root(ka)
    groups={}
    for k in keys:groups.setdefault(root(k),[]).append(tracks[k])
    sem=base['semantic'].copy();conf=base['confidence'].copy();instance=np.zeros(len(sem),np.int32)
    owner=np.zeros(len(sem),np.int32);claims=np.zeros(len(sem),np.int32);ambiguous=np.zeros(len(sem),bool);objects=[]
    fixed=(sem>0)&(sem<33)
    for gid,(groupkey,parts) in enumerate(sorted(groups.items()),1):
        frames=set(f for t in parts for f in t['frames']);ids=np.unique(np.concatenate([np.asarray(t['points'],int) for t in parts]))
        cats={t['category'] for t in parts};fine=cats-{33,34}
        if len(fine)>1:raise ValueError('contradictory fine labels in one group')
        category=next(iter(fine)) if fine else min(cats)
        score=float(np.average([t['mean_point_score'] for t in parts],weights=[len(t['points']) for t in parts]))
        if len(frames)<2 or score<.8 or len(ids)<50:continue
        ids=ids[(~fixed[ids])|(sem[ids]==category)]
        ambiguous[ids]|=(owner[ids]>0)&(owner[ids]!=gid)
        owner[ids]=gid;claims[ids]=category
        objects.append({'instance_id':gid,'semantic_id':category,'source_tracks':[[s,t['track_id']] for s,rows in enumerate(streams) for t in rows if root((s,t['track_id']))==groupkey],
                        'supporting_frames':sorted(frames),'mean_observed_mask_score':score})
    usable=(owner>0)&~ambiguous
    change=usable&~fixed
    sem[change]=claims[change]
    for ob in objects:
        ids=(owner==ob['instance_id'])&usable&(sem==ob['semantic_id']);instance[ids]=ob['instance_id']
        conf[ids&change]=ob['mean_observed_mask_score'];pts=xyz[ids]
        ob['point_count']=len(pts)
        if len(pts):ob.update(center=pts.mean(axis=0).tolist(),min=pts.min(axis=0).tolist(),max=pts.max(axis=0).tolist())
    assert np.array_equal(sem[fixed],base['semantic'][fixed])
    return {'semantic':sem,'instance':instance,'confidence':conf},[o for o in objects if o['point_count']>0],{
      'groups_before':len(keys),'groups_after':len(groups),'retained_groups':sum(o['point_count']>0 for o in objects),
      'ambiguous_ownership_points':int(ambiguous.sum()),'semantic_added_points':int(np.sum((sem>0)&(base['semantic']==0))),
      'coarse_refined_points':int(np.sum((base['semantic']>=33)&(sem<33)&(sem>0))),
      'b_fine_labels_preserved':True,'only_observed_mask_interior_points':True}

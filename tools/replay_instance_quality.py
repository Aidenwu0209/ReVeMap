"""Replay actual fusion from sealed 2D masks and saved projections, without GT.

First reproduce all original labels exactly; then compare policies on identical
inputs. This does not run VLM refinement or certify candidate label accuracy.
"""
import argparse
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

from pathlib import Path
import time
import numpy as np

from pose_pipeline.semantic_runtime.common import read, write, sha
from pose_pipeline.semantic_runtime.fusion import apply_semantic_conflict_policy
from pose_pipeline.semantic_runtime.multiview import fuse_instances
from pose_pipeline.semantic_runtime.object_candidates import build_candidates, extend_candidates_with_pure_masks
from pose_pipeline.semantic_runtime.fragment_merge import merge_fragments
from pose_pipeline.sam3_fusion import MapVotes, GeometricInstances
from pose_pipeline.sam3_refine import interior
from pose_pipeline.sam3_guided import recover_instances
from pose_pipeline.instance_association import select_tracks, compatible, measure_candidates, greedy_pairs, object_consensus


def replay(reference, output):
    reference, output = reference.resolve(strict=True), output.resolve()
    if output == reference or reference in output.parents or output in reference.parents:
        raise ValueError('reference and output must be separate directories')
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    def track(path):
        hashes[str(path)] = sha(path)
        return path
    config = read(track(reference/'CONFIG.json'))
    with np.load(track(reference/'fused/map_labels.npz'), allow_pickle=False) as z:
        original = {k:v.copy() for k,v in z.items()}
    xyz = np.load(track(reference/'fused/target.npz'), allow_pickle=False)['xyz']
    n = len(xyz)
    classes = read(track(reference/'fused/classes.json'))
    votes = MapVotes(n,35); high = np.zeros(n,np.float32)
    streams = [GeometricInstances(),GeometricInstances()]
    frames = []
    started = time.perf_counter()
    for ordinal,row in enumerate(read(track(reference/'semantic/FRAMES.json'))):
        fid = row['frame_id']
        projection = reference/'fused'/f'projection_{fid:06}.npz'
        mask = reference/'semantic/frames'/f'{fid:06}.npz'
        if sha(track(mask)) != row['mask_sha256']:
            raise ValueError('SAM3 mask hash mismatch')
        with np.load(track(projection),allow_pickle=False) as p, np.load(mask,allow_pickle=False) as z:
            ids,v,u = p['point_ids'],p['row'],p['col']
            frame = dict(frame_id=fid,point_ids=ids.copy(),mask_ids=z['local_instance'][v,u],
                         semantic=z['semantic'][v,u],confidence=z['confidence'][v,u],interior=interior(z['semantic'])[v,u])
            if not np.array_equal(p['mask_ids'],frame['mask_ids']) or not np.array_equal(p['semantic'],frame['semantic']):
                raise ValueError('projection does not match original mask')
        safe = frame['interior']
        votes.add(fid,ids,frame['semantic'],frame['confidence'])
        high[ids[safe]] = np.maximum(high[ids[safe]],frame['confidence'][safe])
        streams[ordinal%2].add(fid,ids[safe],frame['mask_ids'][safe],frame['semantic'][safe],frame['confidence'][safe])
        frames.append(frame)
    sem,conf,_ = votes.finalize()
    single = (sem==0)&(votes.counts.sum(1)==1)&(high>=.9)
    sem[single] = votes.scores.argmax(1)[single]; conf[single] = high[single]
    records = [s.records(config.get('semantic_confidence_policy','legacy'),point_scores=high) for s in streams]
    chosen = [select_tracks(r) for r in records]
    pairs = []
    if min(map(len,chosen)) >= 2:
        candidates = [(a['track_id'],b['track_id'],0.) for a in chosen[0] for b in chosen[1] if compatible(a['category'],b['category'])]
        pairs = greedy_pairs(measure_candidates(candidates,*chosen,xyz),'geometry')
    baseline = dict(semantic=sem,confidence=conf)
    guide,_,_ = object_consensus(records,pairs,baseline,xyz)
    guide,_ = apply_semantic_conflict_policy(votes,baseline,guide,config.get('semantic_conflict_policy','consensus'))
    if not np.array_equal(guide['semantic'],original['semantic']):
        raise ValueError('cached semantic replay differs from reference')
    preparation_seconds = time.perf_counter()-started
    variants = {}
    for policy in ('legacy','verified'):
        started = time.perf_counter(); dest=output/policy; dest.mkdir()
        cfg=dict(min_mask_points=30,min_output_points=50,min_point_views=1,
                 min_group_frames=2,object_score_mode='max_point')
        geometry={}
        evidence = {} if policy == 'verified' else None
        current,audit=fuse_instances(n,frames,guide['semantic'],cfg,geometry_out=geometry,
                                     candidate_evidence_out=evidence)
        blocked=[(x['frame_id'],x['mask_id']) for x in audit['filtered_masks']]
        inst,ga,_=recover_instances(n,frames,guide['semantic'],guide['instance'],current,blocked_masks=blocked)
        merge_audit={}
        if policy=='verified':
            inst,merge_audit=merge_fragments(inst,guide['semantic'],frames)
            write(dest/'FRAGMENT_MERGE.json',merge_audit)
        labels={**guide,'instance':inst}
        if policy=='legacy' and any(not np.array_equal(labels[k],original[k]) for k in original):
            raise ValueError('legacy replay did not reproduce all reference labels exactly')
        candidate,ca=build_candidates(labels,geometry)
        if policy == 'verified':
            candidate, expansion = extend_candidates_with_pure_masks(labels, geometry, evidence, candidate, classes)
            ca.update(candidate_points=expansion['candidate_points'], pure_mask_expansion=expansion)
        np.savez_compressed(dest/'map_labels.npz',**labels)
        np.savez_compressed(dest/'candidates.npz',candidate_instance=candidate)
        write(dest/'MULTIVIEW.json',audit);write(dest/'GUIDED.json',ga);write(dest/'CANDIDATES.json',ca)
        variants[policy]=dict(seconds=time.perf_counter()-started,
            changed_instance_points=int(np.sum(inst!=original['instance'])),
            instance_count=len(np.unique(inst[inst>0])),assigned_points=int(np.count_nonzero(inst)),
            candidate_points=ca['candidate_points'],candidate_objects=(len(np.unique(candidate[candidate>0]))
                if policy == 'verified' else len(ca['objects'])),
            fragment_merges=len(merge_audit.get('merges',[])),component_vetoes=merge_audit.get('component_vetoes',0))
        print('VARIANT',reference.parent.name,policy,variants[policy],flush=True)
    if any(sha(path)!=digest for path,digest in hashes.items()):
        raise ValueError('replay inputs changed')
    write(output/'INPUT_HASHES.json',hashes)
    write(output/'SUMMARY.json',dict(reference=str(reference),legacy_exact_parity=True,geometry_modified=False,
        semantic_modified=False,GT_used=False,refinement_inference_executed=False,
        preparation_seconds=preparation_seconds,variants=variants,
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):sha(p)
            for p in sorted((Path(__file__).resolve().parents[1]/'src').rglob('*.py'))}))
    write(output/'PREDICTIONS_LOCK.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    replay(args.reference,args.output)

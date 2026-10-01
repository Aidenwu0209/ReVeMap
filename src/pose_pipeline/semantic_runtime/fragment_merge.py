"""Merge same-class fragments only with repeated shared mask evidence.

Point sets and semantics stay fixed. Every proposed component merge is checked
again using the visible union; missing observations are neither votes nor vetoes.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MergeConfig:
    min_visible_points: int = 30
    containment_fraction: float = .8
    min_support_frames: int = 3
    consensus_fraction: float = .9
    cannot_link_frames: int = 2


def merge_fragments(instance, semantic, frames, config=None):
    cfg = config or MergeConfig()
    if (instance.ndim != 1 or semantic.shape != instance.shape
            or any(not np.issubdtype(a.dtype,np.integer) or np.any(a<0) for a in (instance,semantic))):
        raise ValueError('matching nonnegative instance and semantic IDs required')
    ids = np.unique(instance[instance>0]); n = len(ids)
    # Group once instead of scanning the full map for every instance. Mixed
    # categories still abstain, including mixtures of known and unknown.
    positive = instance > 0
    owners = np.searchsorted(ids, instance[positive])
    low = np.full(n, np.iinfo(semantic.dtype).max, semantic.dtype)
    high = np.zeros(n, semantic.dtype)
    np.minimum.at(low, owners, semantic[positive])
    np.maximum.at(high, owners, semantic[positive])
    categories = np.where(low == high, low, 0)
    observations,seen = [],set()
    for frame in sorted(frames,key=lambda f:f['frame_id']):
        fid=frame['frame_id']
        if fid in seen:
            raise ValueError('duplicate original frame')
        seen.add(fid)
        points,masks=np.asarray(frame['point_ids']),np.asarray(frame['mask_ids'])
        if (points.ndim!=1 or masks.shape!=points.shape
                or not np.issubdtype(points.dtype,np.integer) or not np.issubdtype(masks.dtype,np.integer)
                or np.any(points<0) or np.any(points>=len(instance)) or np.any(masks<0)
                or len(np.unique(points))!=len(points)):
            raise ValueError('invalid projected frame')
        owned=instance[points]>0
        owners=np.searchsorted(ids,instance[points[owned]])
        visible=np.bincount(owners,minlength=n)
        labelled=owned & (masks>0)
        rows=np.searchsorted(ids,instance[points[labelled]])
        _,columns=np.unique(masks[labelled],return_inverse=True)
        # Compact frame-local mask numbers; their identity is only used inside this frame.
        overlap=sparse.coo_matrix((np.ones(len(rows),np.int32),(rows,columns)),
            shape=(n,int(columns.max(initial=-1))+1)).tocsr()
        observations.append((visible,overlap))
    # Singleton signatures dominate the work. Compute all rows of a frame in
    # one sparse pass; reserve repeated sparse slicing for actual component
    # proposals. CSR columns are sorted, so the first maximum retains the old
    # dense argmax tie rule exactly.
    dominant = np.full((n,len(observations)),-1,np.int32)
    for f,(visible,overlap) in enumerate(observations):
        counts = np.diff(overlap.indptr)
        nonempty = np.flatnonzero(counts)
        if not len(nonempty):
            continue
        maxima = np.zeros(n,np.int32)
        maxima[nonempty] = np.maximum.reduceat(overlap.data,overlap.indptr[nonempty])
        rows = np.repeat(np.arange(n),counts)
        best = np.full(n,overlap.shape[1],np.int32)
        is_max = overlap.data == maxima[rows]
        np.minimum.at(best,rows[is_max],overlap.indices[is_max])
        valid = ((visible>=cfg.min_visible_points)
                 & (maxima>=cfg.containment_fraction*visible)
                 & (best<overlap.shape[1]))
        dominant[valid,f] = best[valid]
    signatures={(i,):dominant[i] for i in range(n)}
    def signature(members):
        key=tuple(sorted(members))
        if key not in signatures:
            result=np.full(len(observations),-1,np.int32)
            for f,(visible,overlap) in enumerate(observations):
                total=int(visible[list(key)].sum())
                if total<cfg.min_visible_points or overlap.shape[1]==0:
                    continue
                counts=np.asarray(overlap[list(key)].sum(axis=0)).ravel()
                best=int(counts.argmax())
                if counts[best]>=cfg.containment_fraction*total:
                    result[f]=best
            signatures[key]=result
        return signatures[key]
    def evidence(a,b):
        both=(a>=0)&(b>=0)
        return int(np.sum(both&(a==b))),int(np.sum(both&(a!=b)))
    def passes(p,neg):
        return p>=cfg.min_support_frames and neg<cfg.cannot_link_frames and p/max(1,p+neg)>=cfg.consensus_fraction
    negative=np.zeros((n,n),np.int32);edges=[]
    for a in range(n):
        for b in range(a+1,n):
            # Components only contain one known class, so all cross-class
            # separation counts are unreachable in the component veto.
            if categories[a]<=0 or categories[a]!=categories[b]:
                continue
            p,neg=evidence(dominant[a],dominant[b]);negative[a,b]=negative[b,a]=neg
            if passes(p,neg):
                edges.append((-p,neg,a,b))
    parent=np.arange(n);members={i:{i} for i in range(n)}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    merges=[];vetoes=0
    for _,_,a,b in sorted(edges):
        ra,rb=root(a),root(b)
        if ra==rb:continue
        left,right=members[ra],members[rb]
        p,neg=evidence(signature(left),signature(right))
        if np.any(negative[np.ix_(sorted(left),sorted(right))]>=cfg.cannot_link_frames) or not passes(p,neg):
            vetoes+=1;continue
        lo,hi=sorted((ra,rb));parent[hi]=lo;members[lo]|=members.pop(hi)
        merges.append({'left_instance':int(ids[ra]),'right_instance':int(ids[rb]),
                       'support_frames':p,'separation_frames':neg})
    result=instance.copy();positive=instance>0
    if n:
        mapping=np.array([ids[root(i)] for i in range(n)],dtype=instance.dtype)
        result[positive]=mapping[np.searchsorted(ids,instance[positive])]
    return result,{'method':'same_class_multiview_fragment_merge','config':asdict(cfg),
        'original_instances':n,'merged_instances':len(members),'merges':merges,
        'component_vetoes':vetoes,'GT_used':False,'geometry_modified':False,'semantic_modified':False}

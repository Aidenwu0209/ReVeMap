"""Refine disputed native instance boundaries using independent RGB-D and box-aware queries."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.spatial import cKDTree

from panoptic_observations import CLASSES
from refine_native_panoptic import digest


def load_queries(path, native_count, *, on_mesh):
    with np.load(path, allow_pickle=False) as data:
        classes, scores, masks = (data[k] for k in ("classes","scores","masks"))
        width = int(data["mesh_count"] if on_mesh else data["points"])
        nearest = data["map_to_mesh"] if on_mesh else None
    if scores.shape != classes.shape or classes.ndim != 1 or masks.shape != (len(classes),(width+7)//8):
        raise ValueError(f"Inconsistent query arrays in {path}")
    if not np.isfinite(scores).all() or not ((classes >= 2) & (classes < 20)).all() or masks.dtype != np.uint8:
        raise ValueError(f"Invalid ScanNet object labels, confidence or packed masks in {path}")
    if on_mesh:
        if nearest.shape != (native_count,) or np.any((nearest < 0) | (nearest >= width)):
            raise ValueError(f"Invalid mesh-to-map correspondence in {path}")
    elif width != native_count:
        raise ValueError(f"Query masks have a different native point count in {path}")
    result = []
    for label, score, packed in zip(classes, scores, masks):
        if score >= .34:
            mask = np.unpackbits(packed, count=width).astype(bool)
            result.append((int(label),float(score),mask[nearest] if on_mesh else mask))
    return result


def match_queries(index, classes, queries):
    if not queries or not len(classes):
        return [None] * len(classes)
    counts = np.bincount(index, minlength=len(classes)+1)[1:]
    agreement = np.zeros((len(classes),len(queries)))
    for j,(cls,confidence,mask) in enumerate(queries):
        intersection = np.bincount(index[mask], minlength=len(classes)+1)[1:]
        area = int(mask.sum())
        iou = intersection / np.maximum(1, counts + area - intersection)
        valid = (classes==cls) & (iou>=.20) & (intersection/max(1,area)>=.29)
        agreement[valid,j] = iou[valid] * (.55+.45*confidence)
    matches = [None] * len(classes)
    for i,j in zip(*np.unravel_index(np.argsort(-agreement,axis=None),agreement.shape)):
        if agreement[i,j] <= 0:
            break
        if matches[i] is None and agreement[i,j] >= .90*agreement[:,j].max():
            matches[i] = queries[j][2]
    return matches


def surface_graph(vertex):
    xyz = np.column_stack([vertex[k] for k in "xyz"])
    rgb = np.column_stack([vertex[k] for k in ("red","green","blue")]).astype("f4")/255
    _, index, counts = np.unique(np.floor(xyz/.04).astype("i4"), axis=0, return_inverse=True, return_counts=True)
    index = index.reshape(-1)
    def average(v):
        return np.column_stack([np.bincount(index,weights=channel)/counts for channel in v.T])
    center, color = average(xyz), average(rgb)
    neighbors = min(8,len(center))
    if neighbors < 2:
        return index, counts, (np.empty(0,int),np.empty(0,int),np.empty(0))
    distance, nearest = cKDTree(center).query(center,k=neighbors,workers=2)
    a = np.repeat(np.arange(len(center)),neighbors-1)
    b, distance = nearest[:,1:].ravel(), distance[:,1:].ravel()
    eligible = (a<b) & (distance<.105)
    a, b, distance = a[eligible], b[eligible], distance[eligible]
    color_distance = np.linalg.norm(color[a]-color[b],axis=1)
    weight = np.exp(-.5*(distance/.055)**2-.5*(color_distance/.24)**2)*np.sqrt(np.minimum(counts[a],counts[b]))
    return index, counts, (a,b,weight)


def foreground(support, cells, counts, edges):
    a,b,weight = edges
    lookup = np.full(len(counts),-1,"i4")
    lookup[cells] = np.arange(len(cells))
    left,right = lookup[a],lookup[b]
    inside = (left>=0) & (right>=0)
    left,right,weight = left[inside],right[inside],.42*weight[inside]
    n = len(cells)
    source, sink = n, n+1
    nodes = np.arange(n)
    mass = np.sqrt(counts[cells])
    cost_in = -np.log(np.clip(support,.01,.99))*mass
    cost_out = -np.log(np.clip(1-support,.01,.99))*mass
    rows = np.r_[left,right,np.full(n,source),nodes]
    cols = np.r_[right,left,nodes,np.full(n,sink)]
    values = np.rint(100*np.r_[weight,weight,cost_out,cost_in]).astype("i8")
    graph = coo_matrix((values,(rows,cols)),shape=(n+2,n+2)).tocsr()
    residual = graph - maximum_flow(graph,source,sink).flow
    residual.data[residual.data<=0]=0
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual,source,directed=True,return_predecessors=False)
    keep = np.zeros(n,bool)
    keep[reachable[reachable<n]] = True
    return keep


def refine(vertex, semantic, instance, probabilities, query_sets):
    tracked = (semantic>=2) & (semantic<20) & (instance>0)
    labels, inverse = np.unique(np.column_stack((semantic[tracked],instance[tracked])),axis=0,return_inverse=True)
    index = np.zeros(len(vertex),"i4")
    index[tracked] = inverse.reshape(-1)+1
    matches = [match_queries(index,labels[:,0],queries) for queries in query_sets]
    cell_index, counts, edges = surface_graph(vertex)
    result, ids = semantic.copy(),instance.copy()
    best = np.full(len(vertex),.49,"f4")
    for j,(cls,tag) in enumerate(labels):
        short,long,box = (matched[j] for matched in matches)
        measured = [m for m in (short,long,box) if m is not None]
        if len(measured)<2:
            continue
        native = index==j+1
        region = np.flatnonzero(np.logical_or.reduce([native,*measured]))
        origin = native[region]
        image = np.mean([m[region] for m in (short,long) if m is not None],axis=0)
        bound = box[region] if box is not None else image
        support = .34*origin.astype("f4") + .31*image + .24*bound + .11*probabilities[region,cls]
        cells, local = np.unique(cell_index[region],return_inverse=True)
        local = local.reshape(-1)
        keep = foreground(np.bincount(local,weights=support)/counts[cells],cells,counts,edges)[local]
        gain = keep & ~origin & (image>=.5) & (bound>.5) & (support>best[region])
        loss = ~keep & origin & (image<.5) & (bound<.5) & (best[region]<=.49)
        if gain.sum() > .8*native.sum() or loss.sum() > .55*native.sum():
            continue
        remove = region[loss]
        background = probabilities[remove].argmax(1)
        result[remove] = np.where(background<2,background,20)
        ids[remove] = np.where(background<2,0,-1)
        add = region[gain]
        result[add],ids[add],best[add] = cls,tag,support[gain]
    return result, ids


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {k:digest(getattr(args,k)) for k in ("map","anchor","point_evidence","view19","cap64","box_queries")}
    cloud = PlyData.read(args.map,mmap=False)
    vertex = cloud["vertex"].data
    with np.load(args.anchor,allow_pickle=False) as anchor:
        semantic,instance = anchor["semantic"],anchor["instance"]
    with np.load(args.point_evidence,allow_pickle=False) as data:
        nearest,distance,probability = data["nearest"],data["distance"],data["proba"]
    if not all(v.shape==(len(vertex),) for v in (semantic,instance,nearest,distance)):
        raise ValueError("Anchor and point evidence must use the native map vertex order")
    if probability.ndim!=2 or probability.shape[1]!=20 or np.any((nearest<0)|(nearest>=len(probability))):
        raise ValueError("Expected twenty PTv3 probabilities per mesh vertex")
    probs = probability[nearest].astype("f4")/255
    probs[distance>.12]=0
    queries = [load_queries(args.view19,len(vertex),on_mesh=True),load_queries(args.cap64,len(vertex),on_mesh=True),
               load_queries(args.box_queries,len(vertex),on_mesh=False)]
    semantic,instance = refine(vertex,semantic,instance,probs,queries)
    vertex["semantic_id"],vertex["instance_id"] = semantic.astype("i4")+1,instance
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        dest = Path(temp)/"result"
        dest.mkdir()
        np.savez_compressed(dest/"panoptic.npz",semantic=semantic,instance=instance)
        cloud.write(dest/"map_labeled.ply")
        classes = {0:"unknown", **{i+1:c for i,c in enumerate(CLASSES)}, 21:"unlabeled"}
        (dest/"classes.json").write_text(json.dumps(classes,indent=2))
        report = dict(inputs=inputs, GT_used=False, label_space="ScanNet20", same_native_vertex_order=True,
                      vertex_count=len(vertex), outputs={p.name:digest(p) for p in dest.iterdir()})
        (dest/"result.json").write_text(json.dumps(report,indent=2))
        dest.rename(args.output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("map","anchor","point-evidence","view19","cap64","box-queries","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    run(parser.parse_args())

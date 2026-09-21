"""Secondary diagnostic with class-agnostic recall and background contamination.

The frozen legacy evaluator is run separately without edits. This new diagnostic
uses the same fixed map/GT alignment and 27 eligible things, but counts predicted
mass on other recognized GT categories (including wall/floor) in IoU unions.
It is not official ScanNet AP; unknown GT categories remain outside the domain.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
GT = Path('/Users/wu/Desktop/wu/Xia/jojo_updated_scene0030_run_20260902/source/Jojo_current_panoptic_scene_graph_aligment_on_full_scene0030_00/full_scene0030_00_dataset/gt_scannet_data_scene0030_00')


def read(path):
    return json.loads(path.read_text())


def xyz(vertex):
    return np.stack([vertex[k] for k in ('x', 'y', 'z')], axis=1)


def recall_at(iou, threshold):
    """Maximize threshold-qualified cardinality, then break ties with IoU."""
    if not iou.size:
        return 0.0
    reward = (iou >= threshold).astype(float) * (len(iou) + 1) + iou
    rows, cols = linear_sum_assignment(-reward)
    return float(np.sum(iou[rows, cols] >= threshold) / len(iou))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--objects', type=Path, help='Explicit separately inferred object naming metadata.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    classes_path = ROOT.parent / 'sgf_sga_all_scannet_orbbec_20260910_v1/scenes/scannet/scene0030_00/map/classes.json'
    alignment = ROOT.parent / 'scannet0030_semantic_comparison_20260909_v1/developnew_full/evaluation/diagnostic.json'
    target = ROOT.parent / 'sam3_semantics_20260911_v1/inputs/scene0030_00_target.npz'
    gtfile = GT / 'scene0030_00_vh_clean_2.ply'
    segfile = GT / 'scene0030_00_vh_clean_2.0.010000.segs.json'
    groupfile = GT / 'scene0030_00.aggregation.json'
    objectfile = args.objects or args.labels.parent / 'objects.json'
    paths = [classes_path, alignment, target, gtfile, segfile, groupfile, args.labels, objectfile]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    classes = {value: int(key) for key, value in read(classes_path).items()}
    gx = xyz(PlyData.read(gtfile)['vertex'].data)
    segments = np.asarray(read(segfile)['segIndices'])
    y = np.zeros(len(gx), np.int32)
    g = np.zeros(len(gx), np.int32)
    names = {}
    for item in read(groupfile)['segGroups']:
        points = np.isin(segments, item['segments'])
        gid = int(item['objectId']) + 1
        g[points], y[points] = gid, classes.get(item['label'], 0)
        names[gid] = item['label']
    things = (y > 0) & ~np.isin(y, [10, 19])
    truth_ids = np.array([gid for gid in np.unique(g[things])
                          if np.sum(things & (g == gid)) >= 50], dtype=int)
    eligible = things & np.isin(g, truth_ids)
    assert int(eligible.sum()) == 99812 and len(truth_ids) == 27
    with np.load(target, allow_pickle=False) as data:
        points = data['xyz'].copy()
    with np.load(args.labels, allow_pickle=False) as data:
        instances, semantics = data['instance'].copy(), data['semantic'].copy()
    assert instances.shape == semantics.shape == (len(points),)
    T = np.asarray(read(alignment)['T_dataset_estimated_world'])
    distance, index = cKDTree(points @ T[:3, :3].T + T[:3, 3]).query(gx, workers=1)
    projected = np.where(distance <= .05, instances[index], 0)
    predicted_semantic = semantics[index]
    domain = y > 0
    known = eligible & (projected > 0)
    pred_ids = np.unique(projected[known])
    intersections = np.zeros((len(truth_ids), len(pred_ids)), np.int64)
    if pred_ids.size:
        gi = np.searchsorted(truth_ids, g[known])
        pi = np.searchsorted(pred_ids, projected[known])
        intersections = np.bincount(gi * len(pred_ids) + pi,
                                    minlength=intersections.size).reshape(intersections.shape)
    sizes = np.array([np.sum(eligible & (g == gid)) for gid in truth_ids])
    # Entire recognized GT support of a candidate object counts in its union.
    prediction_sizes = np.array([np.sum(domain & (projected == pid)) for pid in pred_ids])
    unions = sizes[:, None] + prediction_sizes[None, :] - intersections
    iou = np.divide(intersections, unions, out=np.zeros_like(intersections, float), where=unions > 0)
    background = np.isin(y, [10, 19])
    touching = domain & np.isin(projected, pred_ids) & (projected > 0)
    background_mass = int(np.sum(touching & background))
    class_aware_iou = iou.copy()
    predicted_classes = []
    for pid in pred_ids:
        # Include unknown semantic 0. Never invent a semantic label for a mask.
        labels, counts = np.unique(predicted_semantic[domain & (projected == pid)], return_counts=True)
        predicted_classes.append(int(labels[counts.argmax()]))
    for row, gid in enumerate(truth_ids):
        class_aware_iou[row, np.asarray(predicted_classes) != classes[names[int(gid)]]] = 0
    # Actual object-level naming policy from inference, including explicit unknown.
    objects = read(objectfile)
    object_semantics = {int(obj['instance_id']): int(obj['semantic_id']) for obj in objects}
    assert len(object_semantics) == len(objects)
    assert all(int(pid) in object_semantics for pid in pred_ids)
    named_iou = iou.copy()
    exported_classes = np.array([object_semantics[int(pid)] for pid in pred_ids])
    for row, gid in enumerate(truth_ids):
        named_iou[row, exported_classes != classes[names[int(gid)]]] = 0
    per_object = []
    for row, gid in enumerate(truth_ids):
        per_object.append({'gt_object_id': int(gid - 1), 'class': names[int(gid)],
                           'gt_points': int(sizes[row]),
                           'largest_fragment_recall': float(intersections[row].max(initial=0) / sizes[row]),
                           'best_classagnostic_iou': float(iou[row].max(initial=0))})
    matched_background = {}
    for threshold in (.25, .5):
        chosen = np.empty(0, dtype=int)
        if iou.size:
            reward = (iou >= threshold).astype(float) * (len(iou) + 1) + iou
            rr, cc = linear_sum_assignment(-reward)
            chosen = pred_ids[cc[iou[rr, cc] >= threshold]]
        mask = domain & np.isin(projected, chosen) & (projected > 0)
        matched_background[str(threshold)] = {
            'matched_instances': len(chosen),
            'recognized_gt_support': int(mask.sum()),
            'background_points': int(np.sum(mask & background)),
            'background_fraction': float(np.sum(mask & background) / max(1, mask.sum()))}
    result = {
        'protocol': 'secondary_classagnostic_recognized_domain_v1',
        'object_metadata': str(objectfile.resolve()),
        'scope': 'same 27 eligible GT things and 5cm frozen alignment; predicted unions include other recognized objects and wall/floor; not official AP',
        'eligible_gt_points': int(eligible.sum()), 'eligible_gt_objects': len(truth_ids),
        'recognized_gt_domain_points': int(domain.sum()),
        'gt_instance_coverage': float(np.sum(known) / np.sum(eligible)),
        'unknown_semantic_assigned_gt_points': int(np.sum(known & (predicted_semantic == 0))),
        'background_points_in_instances_touching_eligible_things': background_mass,
        'background_contamination_fraction': float(background_mass / max(1, np.sum(touching))),
        'all_touching_metric_caveat': 'Includes structural instances with even tiny overlap on a thing due to geometry mismatch; do not interpret alone as physical object merging rate.',
        'background_in_classagnostically_matched_instances': matched_background,
        'eligible_thing_purity_with_recognized_background_penalty': float(intersections.max(axis=0).sum() / max(1, prediction_sizes.sum())) if pred_ids.size else 0.,
        'classagnostic_recall_iou25': recall_at(iou, .25),
        'classagnostic_recall_iou50': recall_at(iou, .5),
        'posthoc_majority_semantic_recall_iou25': recall_at(class_aware_iou, .25),
        'posthoc_majority_semantic_recall_iou50': recall_at(class_aware_iou, .5),
        'majority_semantic_caveat': 'This diagnostic assigns a label after inference and is not an executed object-naming stage.',
        'exported_object_semantic_recall_iou25': recall_at(named_iou, .25),
        'exported_object_semantic_recall_iou50': recall_at(named_iou, .5),
        'exported_unnamed_instances_touching_things': int(np.sum(exported_classes == 0)),
        'per_object': per_object, 'input_sha256': hashes,
        'gt_used_only_after_inference': True, 'blind_holdout': False,
    }
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('per_object', 'input_sha256')}))


if __name__ == '__main__':
    main()

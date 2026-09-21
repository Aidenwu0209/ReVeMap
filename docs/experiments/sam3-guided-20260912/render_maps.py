"""Fixed full-map comparisons to R2 objects_geometry; display colors never change IDs."""
from pathlib import Path
import argparse
import colorsys
import hashlib
import json
import numpy as np
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
PREVIOUS = R.parent / 'sam3_sga_20260912_v1'


def semantic_colors(labels):
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.stack([50 + labels * m % 206 for m in (73, 151, 199)], axis=-1) / 255.
    colors[labels == 0] = .75
    return colors


def instance_colors(ids):
    ids = np.asarray(ids, dtype=np.int64)
    unique, inverse = np.unique(ids, return_inverse=True)
    palette = np.array([colorsys.hsv_to_rgb((int(i) * .618033988749895) % 1, .66, .96)
                        if i > 0 else (.75, .75, .75) for i in unique])
    return palette[inverse].reshape(ids.shape + (3,))


def instance_display_ids(reference, candidate):
    """Hungarian overlap matching is visualization only, never evaluation or inference."""
    old_ids = np.unique(reference[reference > 0])
    new_ids = np.unique(candidate[candidate > 0])
    mapping = {0: 0}
    pairs = []
    if len(old_ids) and len(new_ids):
        shared = (reference > 0) & (candidate > 0)
        oi = np.searchsorted(old_ids, reference[shared])
        ni = np.searchsorted(new_ids, candidate[shared])
        overlap = np.bincount(oi * len(new_ids) + ni,
                              minlength=len(old_ids) * len(new_ids)).reshape(len(old_ids), len(new_ids))
        old_size = np.bincount(np.searchsorted(old_ids, reference[reference > 0]), minlength=len(old_ids))
        new_size = np.bincount(np.searchsorted(new_ids, candidate[candidate > 0]), minlength=len(new_ids))
        union = old_size[:, None] + new_size[None, :] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)
        for i, j in zip(*linear_sum_assignment(-iou)):
            if iou[i, j] >= .1:
                mapping[int(new_ids[j])] = int(old_ids[i])
                pairs.append({'candidate_id': int(new_ids[j]), 'reference_id': int(old_ids[i]),
                              'whole_map_point_IoU': float(iou[i, j])})
    next_id = int(old_ids.max()) + 1 if len(old_ids) else 1
    for value in new_ids:
        if int(value) not in mapping:
            mapping[int(value)] = next_id
            next_id += 1
    all_new, inverse = np.unique(candidate, return_inverse=True)
    display = np.array([mapping[int(i)] for i in all_new])[inverse]
    return display, {'display_only': True, 'actual_prediction_ids_unchanged': True,
                     'rule': 'one-to-one Hungarian maximum point IoU; reuse reference color only at IoU >= 0.1',
                     'matches': pairs, 'candidate_to_display_id': mapping}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='consensus_instances')
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    run = R / args.stage / args.scene
    reference = PREVIOUS / 'objects_geometry' / args.scene
    vertex = PlyData.read(run / 'map/map_labeled.ply')['vertex'].data
    old = PlyData.read(reference / 'map/map_labeled.ply')['vertex'].data
    xyz = np.stack([vertex[k] for k in ['x', 'y', 'z']], axis=1)
    old_xyz = np.stack([old[k] for k in ['x', 'y', 'z']], axis=1)
    if not np.array_equal(xyz, old_xyz):
        raise ValueError('Comparison requires identical vertex coordinates and order.')
    rgb = np.stack([vertex[k] for k in ['red', 'green', 'blue']], axis=1) / 255.
    candidate_display, color_receipt = instance_display_ids(old['instance_id'], vertex['instance_id'])
    kinds = {
        'semantic': ([rgb, semantic_colors(old['semantic_id']), semantic_colors(vertex['semantic_id'])],
                     ['Original RGB', f'R2 objects_geometry semantic ({np.mean(old["semantic_id"] > 0):.1%} known)',
                      f'Candidate semantic ({np.mean(vertex["semantic_id"] > 0):.1%} known)']),
        'instance': ([rgb, instance_colors(old['instance_id']), instance_colors(candidate_display)],
                     ['Original RGB', f'R2 objects_geometry instances ({np.mean(old["instance_id"] > 0):.1%} assigned)',
                      f'Candidate instances ({np.mean(vertex["instance_id"] > 0):.1%} assigned)'])}
    selected_ids = np.linspace(0, len(xyz) - 1, min(160000, len(xyz)), dtype=int)
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    for kind, (colors, titles) in kinds.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for row, (yaw, pitch) in enumerate([(25, 25), (115, 20)]):
            a, b = np.deg2rad([yaw, pitch])
            ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
            projected = (xyz - center) @ (rx @ ry).T
            order = np.argsort(projected[selected_ids, 2], kind='stable')
            draw = selected_ids[order]
            points = projected[draw]
            bounds = [projected[:, 0].min(), projected[:, 0].max(), projected[:, 1].min(), projected[:, 1].max()]
            for col, color in enumerate(colors):
                axis = axes[row, col]
                axis.scatter(points[:, 0], -points[:, 1], c=color[draw], s=.45, linewidths=0, rasterized=True)
                axis.set_xlim(bounds[0], bounds[1]); axis.set_ylim(-bounds[3], -bounds[2])
                axis.set_aspect('equal'); axis.axis('off'); axis.set_title(titles[col], fontsize=10)
        note = 'Instance colors matched by point overlap for display; actual IDs are unchanged.' if kind == 'instance' else 'Semantic labels are held fixed in the instance-only experiment.'
        fig.suptitle(args.stage + ' / ' + args.scene + '\nSame full geometry, view and sampling. Gray = unknown/unassigned.\n' + note,
                     fontsize=12, x=.5, y=.985)
        # Reserve a fixed heading band for every scene; never let tight_layout move
        # the suptitle into axes. The tight figure bbox includes all text and axes,
        # while unchanged explicit axis limits retain every geometry bound.
        fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.855, hspace=.16, wspace=.035)
        fig.savefig(run / f'map_{kind}_comparison.png', dpi=150, bbox_inches='tight', pad_inches=.15)
        plt.close(fig)
    receipt = {'reference': str(reference), 'map_points': len(xyz), 'display_points': len(selected_ids),
               'shared_selection_all_panels': True, 'identical_original_geometry': True,
               'full_bounds_used': True, 'no_roi_cropping': True,
               'layout': 'fixed heading band, subplot top=0.855; tight figure bbox includes all axes and text',
               'semantic_changed_points': int(np.sum(old['semantic_id'] != vertex['semantic_id'])),
               'cached_sam3_inference_reused': True,
               'display_sampling_sha256': hashlib.sha256(selected_ids.tobytes()).hexdigest(),
               'instance_color_correspondence': color_receipt,
               'figures': ['map_semantic_comparison.png', 'map_instance_comparison.png']}
    (run / 'MAP_VIEW_RECEIPT.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

"""Compare R2/new semantic and instance maps in identical measured Orbbec camera views."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_maps import semantic_colors, instance_colors, instance_display_ids

R = Path(__file__).resolve().parent
B = R.parent / 'sgf_sga_orbbec_4812_20260910_v1'
KEY = 'orbbec/scan_20260909_142829_5ef1fa'
PREVIOUS = R.parent / 'sam3_sga_20260912_v1/objects_geometry' / KEY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam-labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = B / 'metadata' / KEY / 'original_manifest.json'
    trajectory_path = B / 'metadata' / KEY / 'trajectory.json'
    manifest = json.loads(manifest_path.read_text())
    frames = {f['frame_id']: f for f in manifest['frames']}
    poses = {p['frame_id']: np.array(p['T_world_camera_m']).reshape(4, 4)
             for p in json.loads(trajectory_path.read_text())['poses']}
    vertex = PlyData.read(B / 'baseline/refused.ply')['vertex'].data
    xyz = np.stack([vertex[k] for k in ['x', 'y', 'z']], axis=1)
    with np.load(PREVIOUS / 'map_labels.npz') as data:
        old_sem, old_inst = data['semantic'].copy(), data['instance'].copy()
    with np.load(args.sam_labels) as data:
        new_sem, new_inst = data['semantic'].copy(), data['instance'].copy()
    if any(len(v) != len(xyz) for v in [old_sem, old_inst, new_sem, new_inst]):
        raise ValueError('Map label counts differ from original geometry.')
    xyz_hash = hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()
    candidate_report = json.loads((args.sam_labels.parent / 'result.json').read_text())
    if candidate_report.get('geometry_xyz_sha256') != xyz_hash:
        raise ValueError('Candidate result does not reference the exact original geometry.')
    display, color_receipt = instance_display_ids(old_inst, new_inst)
    colors = [semantic_colors(old_sem), semantic_colors(new_sem),
              instance_colors(old_inst), instance_colors(display)]
    labels = [old_sem, new_sem, old_inst, new_inst]
    names = ['reference_semantic', 'candidate_semantic', 'reference_instance', 'candidate_instance']
    titles = ['Raw RGB', 'R2 objects_geometry semantic', 'Candidate semantic',
              'R2 objects_geometry instances', 'Candidate instances']
    selected_frames = [1000, 3000, 4811]
    fig, axes = plt.subplots(3, 5, figsize=(22, 11))
    rows = []
    for row, fid in enumerate(selected_frames):
        rgb = np.array(Image.open(B / 'camera_view_audit' / f'{fid}_color_path.png').convert('RGB'))
        depth = np.array(Image.open(B / 'camera_view_audit' / f'{fid}_depth_path.png')) / manifest['depth_scale']
        height, width = depth.shape
        transform = poses[fid]
        fx, fy, cx, cy = frames[fid]['intrinsics']
        camera = (xyz - transform[:3, 3]) @ transform[:3, :3]
        indices = np.flatnonzero((camera[:, 2] > .1) & (camera[:, 2] <= 4.5))
        points = camera[indices]
        u = np.rint(fx * points[:, 0] / points[:, 2] + cx).astype(int)
        v = np.rint(fy * points[:, 1] / points[:, 2] + cy).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices, u, v = indices[inside], u[inside], v[inside]
        z = camera[indices, 2]
        pixels = v * width + u
        order = np.argsort(z, kind='stable')
        _, first = np.unique(pixels[order], return_index=True)
        keep = order[first]
        indices, u, v, z = indices[keep], u[keep], v[keep], z[keep]
        valid = (depth[v, u] > 0) & (np.abs(depth[v, u] - z) <= .05)
        indices, u, v = indices[valid], u[valid], v[valid]
        images = [rgb]
        stats = {'frame': fid, 'same_visible_map_pixels': len(indices)}
        for name, label, color in zip(names, labels, colors):
            visible_labels = label[indices]
            known = visible_labels > 0
            overlay = rgb.copy()
            overlay[v[known], u[known]] = (.15 * rgb[v[known], u[known]] +
                                          .85 * 255 * color[indices[known]]).astype('u1')
            images.append(overlay)
            stats[name] = {'known_visible_pixels': int(known.sum()),
                           'distinct_nonzero_labels': int(len(np.unique(visible_labels[known])))}
        rows.append(stats)
        for col, image in enumerate(images):
            axes[row, col].imshow(image)
            axes[row, col].axis('off')
            axes[row, col].set_title(titles[col] + f' / {fid}', fontsize=10)
    fig.suptitle(args.sam_labels.parts[-4] + ' / Orbbec: same original cameras, vertices and 5 cm depth gate\n'
                 'Sparse measured map point centres only; no hole filling. Unknown/unassigned pixels retain RGB.\n'
                 'Instance colors matched by map overlap for display; IDs and predictions are unchanged.', fontsize=12)
    fig.subplots_adjust(left=.005, right=.995, top=.88, bottom=.015, wspace=.02, hspace=.10)
    fig.savefig(args.output / 'camera_comparison.png', dpi=140)
    plt.close(fig)
    receipt = {'rows': rows, 'scope': 'visual audit only; Orbbec has no semantic/instance GT here',
               'reference_labels': str(PREVIOUS / 'map_labels.npz'),
               'candidate_labels': str(args.sam_labels), 'no_hole_filling': True,
               'identical_original_geometry': True, 'geometry_xyz_sha256': xyz_hash,
               'depth_tolerance_m': .05, 'semantic_labels_identical': bool(np.array_equal(old_sem, new_sem)),
               'cached_sam3_inference_reused': True, 'instance_color_correspondence': color_receipt}
    (args.output / 'CAMERA_COMPARISON.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()

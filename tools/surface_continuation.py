"""Select measured additions that extend locally supported surfaces of a ReVeMap map."""

import numpy as np
from scipy.spatial import cKDTree


def continuation_mask(primary_xyz, added_xyz, primary_semantic, added_semantic, policy="tight"):
    """Check whether an added point is near the primary surface or extends its tangent plane."""
    primary = np.asarray(primary_xyz, dtype=np.float32)
    added = np.asarray(added_xyz, dtype=np.float32)
    if policy == "tight":
        near_distance, max_distance, offset_intercept, slope = 0.055, 0.18, 0.015, 0.12
    elif policy == "relaxed":
        near_distance, max_distance, offset_intercept, slope = 0.070, 0.30, 0.025, 0.15
    else:
        raise ValueError("Surface policy must be 'tight' or 'relaxed'")
    if not len(added):
        return np.empty(0, dtype=bool)
    tree = cKDTree(primary)
    distance, nearest = tree.query(added)
    keep = distance <= near_distance
    if len(primary) < 18:
        return keep

    # Fit at each distinct primary anchor, since several additions may share one.
    anchors, lookup = np.unique(nearest, return_inverse=True)
    normals = np.empty((len(anchors), 3), dtype=np.float32)
    roughness = np.empty(len(anchors), dtype=np.float32)
    radius = np.empty(len(anchors), dtype=np.float32)
    for first in range(0, len(anchors), 4096):
        batch = slice(first, first + 4096)
        distances, neighbors = tree.query(primary[anchors[batch]], k=18)
        points = primary[neighbors].astype(np.float64)
        centered = points - points.mean(axis=1, keepdims=True)
        covariance = centered.transpose(0, 2, 1) @ centered / points.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normals[batch] = eigenvectors[:, :, 0]
        roughness[batch] = np.sqrt(np.maximum(eigenvalues[:, 0], 0))
        radius[batch] = distances[:, -1]

    offset = np.abs(np.einsum("ij,ij->i", added - primary[nearest], normals[lookup]))
    same_class = added_semantic == primary_semantic[nearest]
    stable_plane = (roughness[lookup] < 0.017) & (radius[lookup] < 0.14)
    return keep | ((distance <= max_distance) & same_class & stable_plane & (offset <= offset_intercept + slope * distance))


def select_additions(primary_xyz, added_xyz, primary_semantic, added_semantic,
                     primary_instances, added_instances, depth_keep, policy="tight"):
    """Apply both geometry checks to additions outside an established primary instance."""
    if (depth_keep.dtype != np.bool_ or depth_keep.shape != (len(added_xyz),)
            or primary_semantic.shape != primary_instances.shape
            or primary_semantic.shape != (len(primary_xyz),)
            or added_semantic.shape != added_instances.shape or added_semantic.shape != depth_keep.shape):
        raise ValueError("Point labels and depth decisions must follow the original vertex order")
    continuation = continuation_mask(primary_xyz, added_xyz, primary_semantic, added_semantic, policy)
    primary_ids = np.unique(primary_instances[primary_instances > 0])
    established = (added_instances > 0) & np.isin(added_instances, primary_ids)
    return established | (continuation & depth_keep)

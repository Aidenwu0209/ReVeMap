"""Estimate metric room axes from the reconstructed mesh and camera poses."""
import numpy as np


def camera_normal_frame(poses, vertices, triangles):
    """Reproduce the single-expert rotation used on ORB-half reconstructions."""
    points = vertices.astype(np.float32)
    up = norm(np.median(-poses[:, :3, 1], axis=0))
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(area[:, None], 1e-20)
    for degrees in (30, 18, 12):
        agreement = normals @ up
        selected = (np.abs(agreement) > np.cos(np.deg2rad(degrees))) & (area > 0)
        if not selected.any():
            break
        up = norm(np.sum(cross[selected] * np.sign(agreement[selected, None]), axis=0))
    right = poses[0, :3, 0] - up * np.dot(poses[0, :3, 0], up)
    if np.linalg.norm(right) < .1:
        axis = np.eye(3)[np.argmin(np.abs(up))]
        right = axis - up * np.dot(axis, up)
    right = norm(right)
    return np.stack([right, np.cross(up, right), up]), {"method": "camera-normal"}

def norm(v):
    return v / np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
def axis_frame(T, vertices, triangles):
    up=norm(-T[::max(1,len(T)//100),:3,1])
    camera_up=norm(np.median(up,axis=0))
    v=vertices[triangles]
    cross=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]); mass=np.linalg.norm(cross,axis=1)
    normals=norm(cross)
    dots=normals@camera_up
    ok=(np.abs(dots)>np.cos(np.deg2rad(30)))&(mass>1e-8)
    height=normals[ok]*np.sign(dots[ok,None]);weight=mass[ok]
    chosen=camera_up
    surface_fraction=0.
    if len(weight):
        # Map normal components to orientation bins and use the 20 largest area bins as candidates.
        _,ix=np.unique(np.rint(height/.08).astype(np.int16),axis=0,return_inverse=True)
        bin_mass=np.bincount(ix,weights=weight)
        centers=norm(np.stack([np.bincount(ix,weights=height[:,j]*weight) for j in range(3)],1))
        rankings=np.argsort(-bin_mass,kind='stable')[:20]
        best=0.
        for seed in centers[rankings]:
            sel=(height@seed)>np.cos(np.deg2rad(7))
            total=weight[sel].sum()
            if total>best:
                chosen=norm((height[sel]*weight[sel,None]).sum(0))
                best=total
        surface_fraction=float(best/mass.sum())
        if surface_fraction<.025:chosen=camera_up
    upright=norm(chosen)
    heading=T[:,:3,2] - (T[:,:3,2]@upright)[:,None]*upright
    good=np.flatnonzero(np.linalg.norm(heading,axis=1)>.2)
    if len(good):
        y=norm(heading[good[0]]);x=norm(np.cross(y,upright))
    else:
        camera_right=T[0,:3,0]
        x=norm(camera_right-upright*(camera_right@upright))
    y=norm(np.cross(upright,x))
    R=np.stack([x,y,upright])
    np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-6)
    assert np.linalg.det(R)>.9999
    return R,dict(camera_up=camera_up.tolist(),chosen_native_up=upright.tolist(),
                  consensus_surface_fraction=surface_fraction,mesh_used=surface_fraction>=.025,
                  angle_to_camera_prior_deg=float(np.rad2deg(np.arccos(np.clip(camera_up@upright,-1,1)))))

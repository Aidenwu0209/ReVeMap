"""Reproduce the archived ScanNet depth-camera RGB, without reading GT poses.

This experiment adapter is deliberately strict: it supports the identical
color/depth extrinsics present in these .sens files and fails otherwise.
Other datasets keep their existing read_frame contract.
"""
from pathlib import Path
import hashlib, struct
import numpy as np


class RegisteredInput:
    def __init__(self, scene, spec):
        self.scene = scene
        self.audit = {'scene': scene, 'GT_pose_decoded': False}
        self.offsets = None
        if spec.get('rgb_registration') == 'already_registered':
            self.audit['mode'] = 'existing_manifest_rgbd_contract'
            return
        if spec.get('rgb_registration') != 'scannet_sens':
            raise ValueError('INPUT.json must declare rgb_registration: already_registered or scannet_sens')
        import cv2
        self.path = Path(spec['sens_path']).resolve(strict=True)
        self.offsets = []
        def unpack(f, fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
        with self.path.open('rb') as f:
            assert unpack(f, 'I')[0] == 4
            f.seek(unpack(f, 'Q')[0], 1)
            kc, ec, kd, ed = [np.asarray(unpack(f, '16f')).reshape(4, 4) for _ in range(4)]
            color_codec, depth_codec = unpack(f, 'ii')
            cw, ch, dw, dh = unpack(f, 'IIII')
            scale = unpack(f, 'f')[0]
            count = unpack(f, 'Q')[0]
            assert color_codec == 2 and depth_codec == 1
            assert np.allclose(ec, ed) and np.allclose(ec, np.eye(4)), 'nontrivial RGB-D extrinsics require depth-dependent registration'
            self.header_end = f.tell()
            for fid in range(count):
                f.seek(80, 1)  # Skip 64-byte GT pose and 16-byte timestamps, never decode.
                nc, nd = unpack(f, 'QQ')
                self.offsets.append((f.tell(), nc))
                f.seek(nc + nd, 1)
            trailing_bytes = self.path.stat().st_size - f.tell()
            imu_count = unpack(f, 'Q')[0] if trailing_bytes else 0
            assert trailing_bytes == 0 or trailing_bytes == 8 + 128 * imu_count
        self.kd = kd[:3, :3]
        yy, xx = np.mgrid[:dh, :dw]
        rays = np.stack([xx.ravel(), yy.ravel(), np.ones(dw * dh)])
        q = kc[:3, :3] @ np.linalg.inv(kd[:3, :3]) @ rays
        self.mx = (q[0] / q[2]).reshape(dh, dw).astype('f4')
        self.my = (q[1] / q[2]).reshape(dh, dw).astype('f4')
        with self.path.open('rb') as f:
            header_hash = hashlib.sha256(f.read(self.header_end)).hexdigest()
        self.audit.update(mode='original_sens_jpeg_calibrated_remap', source=str(self.path),
            source_bytes=self.path.stat().st_size, header_sha256=header_hash,
            intrinsic_color=kc.tolist(), intrinsic_depth=kd.tolist(),
            color_size=[cw, ch], output_depth_size=[dw, dh], depth_scale=scale,
            frame_count=count, skipped_imu_records=imu_count, cv2_version=cv2.__version__, used_color_payload_sha256={})

    def read(self, frame):
        from pose_pipeline.sam3_mapping import read_frame
        image, depth, intrinsics = read_frame(frame)
        if self.offsets is None:
            return image, depth, intrinsics
        import cv2
        from PIL import Image
        assert not frame.rotate_ccw
        assert depth.shape == self.mx.shape
        fx, fy, cx, cy = intrinsics
        assert np.allclose([fx, fy, cx, cy], [self.kd[0, 0], self.kd[1, 1], self.kd[0, 2], self.kd[1, 2]], atol=1e-4)
        offset, size = self.offsets[frame.frame_id]
        with self.path.open('rb') as f:
            f.seek(offset)
            payload = f.read(size)
        assert len(payload) == size
        self.audit['used_color_payload_sha256'][str(frame.frame_id)] = hashlib.sha256(payload).hexdigest()
        bgr = cv2.imdecode(np.frombuffer(payload, dtype='u1'), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.remap(bgr, self.mx, self.my, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), depth, intrinsics

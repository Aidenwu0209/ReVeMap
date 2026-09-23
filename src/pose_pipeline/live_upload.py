"""Durable, resumable storage of local Scan protocol-v1 packets (no GPU work)."""
import hashlib
import json
import os
from pathlib import Path
import re
import threading

MAX_MANIFEST = 4 * 1024 * 1024
MAX_PACKET = 32 * 1024 * 1024


def durable_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class UploadStore:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()

    def directory(self, upload_id):
        if not re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', upload_id):
            raise ValueError('Invalid upload id')
        return self.root / upload_id.lower()

    def manifest(self, upload_id):
        return json.loads((self.directory(upload_id) / 'manifest.json').read_text())

    def prepare(self, body):
        if not isinstance(body, dict):
            raise ValueError('Expected manifest object')
        upload_id = body.get('id', '')
        root = self.directory(upload_id)
        frames = body.get('frames')
        if not isinstance(frames, list) or not 1 <= len(frames) <= 30000:
            raise ValueError('Expected 1–30000 frames')
        normalized = []
        for frame in frames:
            if (not isinstance(frame, dict) or type(frame.get('bytes')) is not int
                    or not 12 <= frame['bytes'] <= MAX_PACKET
                    or not isinstance(frame.get('sha256'), str)
                    or not re.fullmatch('[0-9a-f]{64}', frame['sha256'])):
                raise ValueError('Invalid frame length or digest')
            normalized.append({'bytes': frame['bytes'], 'sha256': frame['sha256']})
        digest = hashlib.sha256(''.join(f['sha256']+'\n' for f in normalized).encode()).hexdigest()
        manifest = {'id': upload_id.lower(), 'frames': normalized, 'sha256': digest}
        with self.lock:
            root.mkdir(parents=True, exist_ok=True)
            path = root / 'manifest.json'
            if path.exists():
                if json.loads(path.read_text()) != manifest:
                    raise ValueError('Upload id already belongs to different data')
            else:
                durable_json(path, manifest)
            missing = [i for i, f in enumerate(normalized) if not self._matches(root, i, f)]
            return {'id': manifest['id'], 'sha256': digest, 'missing': missing}

    @staticmethod
    def _matches(root, index, frame):
        path = root / f'f{index:06d}.bin'
        return (path.is_file() and path.stat().st_size == frame['bytes']
                and file_digest(path) == frame['sha256'])

    def put(self, upload_id, index, stream, length):
        with self.lock:
            root = self.directory(upload_id)
            manifest = self.manifest(upload_id)
            if not 0 <= index < len(manifest['frames']):
                raise ValueError('Invalid frame index')
            frame = manifest['frames'][index]
            if length != frame['bytes']:
                raise ValueError('Frame length mismatch')
            path = root / f'f{index:06d}.bin'
            tmp = path.with_suffix('.part')
            digest = hashlib.sha256()
            remaining = length
            magic = b''
            try:
                with tmp.open('wb') as f:
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError('Incomplete frame; retry this frame')
                        if len(magic) < 8:
                            magic += block[:8-len(magic)]
                        f.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    if magic != b'SGFIPD01' or digest.hexdigest() != frame['sha256']:
                        raise ValueError('Frame checksum or protocol magic mismatch')
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
                # Persist the directory entry before acknowledging the frame.
                fd = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                tmp.unlink(missing_ok=True)
            return {'index': index, 'sha256': frame['sha256']}

    def seal(self, upload_id):
        with self.lock:
            root = self.directory(upload_id)
            manifest = self.manifest(upload_id)
            if any(not self._matches(root, i, f) for i, f in enumerate(manifest['frames'])):
                raise ValueError('Missing or corrupt frames; resume upload first')
            receipt = {'id': manifest['id'], 'confirmed': True,
                       'frames': len(manifest['frames']),
                       'bytes': sum(f['bytes'] for f in manifest['frames']),
                       'sha256': manifest['sha256']}
            durable_json(root / 'receipt.json', receipt)
            return receipt

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid


from pose_pipeline.live_upload import UploadStore


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = UploadStore(Path(self.temp.name))
        self.uid = str(uuid.uuid4())
        self.packets = [b'SGFIPD01' + bytes([i])*100 for i in range(3)]
        self.manifest = {'id': self.uid, 'frames': [
            {'bytes': len(p), 'sha256': hashlib.sha256(p).hexdigest()} for p in self.packets]}

    def tearDown(self):
        self.temp.cleanup()

    def put(self, i):
        return self.store.put(self.uid, i, io.BytesIO(self.packets[i]), len(self.packets[i]))

    def test_restart_resume_and_receipt(self):
        self.assertEqual(self.store.prepare(self.manifest)['missing'], [0,1,2])
        self.put(0)
        self.store = UploadStore(Path(self.temp.name))
        self.assertEqual(self.store.prepare(self.manifest)['missing'], [1,2])
        self.put(1); self.put(2)
        receipt = self.store.seal(self.uid)
        self.assertEqual(receipt['frames'], 3)
        self.assertEqual(receipt['bytes'], 324)
        self.assertTrue(receipt['confirmed'])
        self.assertEqual(self.store.prepare(self.manifest)['missing'], [])
        self.assertEqual(self.store.seal(self.uid), receipt)

    def test_truncated_packet_retry(self):
        self.store.prepare(self.manifest)
        with self.assertRaises(ValueError):
            self.store.put(self.uid, 0, io.BytesIO(self.packets[0][:20]), len(self.packets[0]))
        self.assertFalse(list(self.store.directory(self.uid).glob('*.part')))
        self.assertEqual(self.store.prepare(self.manifest)['missing'], [0,1,2])
        self.put(0)

    def test_corrupt_packet_rejected_and_no_receipt(self):
        self.store.prepare(self.manifest)
        with self.assertRaises(ValueError):
            self.store.put(self.uid, 0, io.BytesIO(b'x'*108), 108)
        with self.assertRaises(ValueError): self.store.seal(self.uid)
        self.assertFalse((self.store.directory(self.uid)/'receipt.json').exists())

    def test_post_write_corruption_detected(self):
        self.store.prepare(self.manifest)
        for i in range(3): self.put(i)
        (self.store.directory(self.uid)/'f000001.bin').write_bytes(b'x'*108)
        self.assertEqual(self.store.prepare(self.manifest)['missing'], [1])
        with self.assertRaises(ValueError): self.store.seal(self.uid)

    def test_duplicate_id_cannot_change_content(self):
        self.store.prepare(self.manifest)
        self.manifest['frames'][0]['sha256'] = '0'*64
        with self.assertRaises(ValueError): self.store.prepare(self.manifest)

    def test_disk_failure_does_not_acknowledge(self):
        self.store.prepare(self.manifest)
        with patch('pose_pipeline.live_upload.os.fsync', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError): self.put(0)
        self.assertEqual(self.store.prepare(self.manifest)['missing'], [0,1,2])

    def test_retransmit_is_idempotent(self):
        self.store.prepare(self.manifest)
        self.assertEqual(self.put(0), self.put(0))
        self.assertEqual(len(list(self.store.directory(self.uid).glob('*.bin'))), 1)

    def test_bad_ids_lengths_and_indices(self):
        with self.assertRaises(ValueError): self.store.directory('../scan_x')
        with self.assertRaises(ValueError): self.store.prepare({'id':self.uid,'frames':[]})
        self.store.prepare(self.manifest)
        with self.assertRaises(ValueError): self.store.put(self.uid,-1,io.BytesIO(),0)
        with self.assertRaises(ValueError): self.store.put(self.uid,0,io.BytesIO(),0)


if __name__ == '__main__': unittest.main(verbosity=2)

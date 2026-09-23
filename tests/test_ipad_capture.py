import os
import hashlib
import io
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import uuid
import zlib
import cv2
import numpy as np
from pose_pipeline.device_capture import packet_frames
from pose_pipeline.device_gui import Controller
from pose_pipeline.live_upload import UploadStore


def packet(i):
    _, jpeg = cv2.imencode('.jpg', np.full((32,32,3),100+i,dtype=np.uint8))
    payloads={'color':jpeg.tobytes(),'depth':np.full((16,16),1.5,dtype='<f4').tobytes(),'confidence':bytes([2])*256}
    header={'schema_version':1,'image_orientation':'sensor_native','color_encoding':'jpeg',
            'depth_encoding':'float32_le_meters','confidence_encoding':'uint8',
            'color_width':32,'color_height':32,'depth_width':16,'depth_height':16,
            'intrinsics_reference_width':32,'intrinsics_reference_height':32,
            'frame_id':i,'timestamp_ns':1000000000+i*100000000,
            'camera_intrinsics':[32.+i*.01,0,16,0,32.+i*.01,16,0,0,1],
            'arkit_camera_to_world_m':np.eye(4).reshape(-1).tolist(),
            'session_id':'synthetic-test-only','tracking_state':'normal',
            'crc32':{k:zlib.crc32(v)&0xffffffff for k,v in payloads.items()},
            **{k+'_bytes':len(v) for k,v in payloads.items()}}
    data=json.dumps(header).encode()
    return b'SGFIPD01'+struct.pack('>I',len(data))+data+b''.join(payloads.values())


class CaptureUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=UploadStore(self.root/'uploads');self.uid=str(uuid.uuid4())
        self.packets=[packet(i) for i in range(3)]
        self.store.prepare({'id':self.uid,'frames':[{'bytes':len(p),'sha256':hashlib.sha256(p).hexdigest()} for p in self.packets]})
        for i,p in enumerate(self.packets):self.store.put(self.uid,i,io.BytesIO(p),len(p))
        self.store.seal(self.uid)
    def tearDown(self):self.tmp.cleanup()
    def test_real_decoder_preserves_native_grid_and_locked_intrinsics(self):
        rows=list(packet_frames(self.store.directory(self.uid)))
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0][0].shape,(32,32,3))
        self.assertEqual(rows[0][1].shape,(32,32))
        self.assertTrue(all(row[2]==rows[0][2] for row in rows))
        self.assertTrue(all((row[1]==1500).all() for row in rows))
        self.assertEqual([row[4]['source_frame_id'] for row in rows],[0,1,2])
    def test_capture_cli_seals_all_frames(self):
        session=self.root/'capture-test';session.mkdir()
        subprocess.run([sys.executable,'-m','pose_pipeline.device_capture','--session',str(session),
                        '--packet-dir',str(self.store.directory(self.uid))],check=True,timeout=30,env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')})
        status=json.loads((session/'capture_status.json').read_text())
        self.assertEqual(status['status'],'sealed');self.assertEqual(status['frames'],3)
        manifest=json.loads((session/'capture/manifest.json').read_text())
        self.assertIn('ipad_lidar_verified_local_upload',json.dumps(manifest))
    def test_standby_accepts_fragmented_magic_but_ignores_empty_probe(self):
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        c=Controller(SimpleNamespace(output=self.root/'sessions',replay=None,ipad_port=port,ipad_bind='127.0.0.1',wireless_host=None))
        started=threading.Event()
        def start(options=None):started.set();c.shutdown.set()
        c.start=start
        worker=threading.Thread(target=c.ipad_standby_loop,daemon=True);worker.start()
        deadline=time.monotonic()+3
        while c.state.get('standby')!='listening' and time.monotonic()<deadline:time.sleep(.01)
        with socket.create_connection(('127.0.0.1',port),timeout=2):pass
        time.sleep(.05);self.assertFalse(started.is_set())
        with socket.create_connection(('127.0.0.1',port),timeout=2) as client:
            for chunk in (b'SG',b'FIP',b'D01'):
                client.sendall(chunk);time.sleep(.05)
        self.assertTrue(started.wait(2));worker.join(3);self.assertFalse(worker.is_alive())

if __name__=='__main__':unittest.main(verbosity=2)

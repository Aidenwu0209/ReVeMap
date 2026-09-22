"""Real POSIX watchdog lifetimes; no cameras or model output are substituted."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def running(pid):
    state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith('Z')


def wait_for(predicate, timeout=6):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.02)
    pytest.fail('watchdog process tree did not reach its expected state')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process groups')
@pytest.mark.parametrize('trigger', ['parent_sigkill', 'owner_sigkill', 'owner_exit', 'fork_owner_sigkill'])
def test_detached_monitor_reaps_uninstrumented_descendant(tmp_path, trigger):
    script = tmp_path / 'watchdog_tree.py'
    script.write_text('''import os,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);role=sys.argv[2];forked=sys.argv[3]=='fork'
if role=='parent':
 child=subprocess.Popen([sys.executable,__file__,str(root),'owner',sys.argv[3]],
  env={**os.environ,'REVEMAP_SUPERVISOR_PID':str(os.getpid())},start_new_session=True)
 (root/'owner.pid').write_text(str(child.pid))
 (root/'owner.returncode').write_text(str(child.wait()))
 while True:time.sleep(.02)
elif role=='owner':
 import pose_pipeline.live_io as io
 io.guard_parent_process(interval=.02,grace=.2)
 monitor=io._PARENT_GUARD[2]
 # Re-entering the same hook must not create another detached monitor.
 io.guard_parent_process(interval=.02,grace=.2)
 assert io._PARENT_GUARD[2].pid==monitor.pid
 (root/'monitor.pid').write_text(str(monitor.pid))
 if forked:
  pid=os.fork()
  if pid==0:
   signal.signal(signal.SIGTERM,signal.SIG_IGN)
   (root/'leaf.pid').write_text(str(os.getpid()))
   while True:time.sleep(.02)
 else:
  subprocess.Popen([sys.executable,__file__,str(root),'leaf','plain'])
 while not (root/'owner_exit').exists():time.sleep(.02)
 # A normal interpreter exit also closes its pipe and reaps the leftover group.
 sys.exit(0)
else:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 (root/'leaf.pid').write_text(str(os.getpid()))
 while True:time.sleep(.02)
''')
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    env.pop('REVEMAP_SUPERVISOR_PID', None); env.pop('REVEMAP_GUI_PARENT_PID', None)
    parent = subprocess.Popen([sys.executable, str(script), str(tmp_path), 'parent',
                               'fork' if trigger.startswith('fork') else 'plain'],
                              env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pids = {}
    try:
        wait_for(lambda: all((tmp_path / f'{name}.pid').exists() for name in ('owner', 'monitor', 'leaf')))
        pids = {name: int((tmp_path / f'{name}.pid').read_text()) for name in ('owner', 'monitor', 'leaf')}
        assert os.getpgid(pids['owner']) == os.getpgid(pids['leaf']) == pids['owner']
        assert os.getpgid(pids['monitor']) == pids['monitor']
        if trigger == 'parent_sigkill':
            parent.kill(); parent.wait(timeout=3)
        elif trigger == 'owner_exit':
            (tmp_path / 'owner_exit').touch()
        else:
            os.kill(pids['owner'], signal.SIGKILL)
        wait_for(lambda: all(not running(pid) for pid in pids.values()))
        if trigger == 'owner_exit':
            assert (tmp_path / 'owner.returncode').read_text() == '0'
    finally:
        if parent.poll() is None: parent.kill(); parent.wait(timeout=3)
        for pid in pids.values():
            if running(pid):
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process groups')
def test_normal_worker_exit_code_remains_zero_twenty_times(tmp_path):
    # The earlier thread/EOF-only monitors passed quick exits but killed this
    # perfectly normal atexit cleanup with -SIGTERM in all 20 reproduced runs.
    code = '''import atexit,time
atexit.register(time.sleep,.1)
import pose_pipeline.live_io as io
io.guard_parent_process(interval=.02,grace=.2)
print(io._PARENT_GUARD[2].pid,flush=True)
time.sleep(.03)
'''
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
           'PYTHONDONTWRITEBYTECODE': '1', 'REVEMAP_SUPERVISOR_PID': str(os.getpid()),
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    monitors, codes = [], []
    try:
        for _ in range(20):
            result = subprocess.run([sys.executable, '-c', code], env=env, start_new_session=True,
                                    text=True, capture_output=True, timeout=5)
            monitors.append(int(result.stdout.strip()))
            codes.append(result.returncode)
        assert codes == [0] * 20
        wait_for(lambda: all(not running(pid) for pid in monitors))
    finally:
        for pid in monitors:
            if running(pid):
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass

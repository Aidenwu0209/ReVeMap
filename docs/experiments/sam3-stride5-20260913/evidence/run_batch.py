from pathlib import Path
import os,subprocess,json,time,traceback,hashlib
R=Path(__file__).resolve().parent
GPU='/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python'
CPU='/home/aidenwu/Documents/sgf_sga_restore_20260910_v1/sga_env/bin/python'
def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def main():
 plan=json.loads((R/'PLAN.json').read_text());done=[];start=time.time()
 for j in plan['jobs']:
  d=R/j['key'];env={**os.environ,'EXPERIMENT_SCENE_DIR':str(d),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
  with (d/'INFERENCE.log').open('w') as log:
   p=subprocess.Popen([GPU,'-u',str(R/'infer_frames.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
   try:
    for stride in plan['strides']:
     while not (d/f'READY_stride{stride}.json').exists():
      if p.poll() is not None:raise RuntimeError(f'inference exited {p.returncode} before ready {j["key"]} {stride}')
      time.sleep(5)
     write('STATUS.json',{'phase':'backend','scene':j['key'],'stride':stride,'done':done,'elapsed_seconds':time.time()-start})
     with (d/f'REPLAY_stride{stride}.log').open('w') as out:
      subprocess.run([CPU,'-u',str(R/'replay_arm.py'),'--stride',str(stride)],env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
     done.append([j['key'],stride]);write('STATUS.json',{'phase':'inference_or_next_arm','scene':j['key'],'done':done,'elapsed_seconds':time.time()-start})
    if p.wait()!=0:raise RuntimeError('inference final verification failed')
   finally:
    if p.poll() is None:p.terminate();p.wait(timeout=20)
 write('COMPLETE.json',{'status':'completed','done':done,'wall_seconds':time.time()-start,'quality_accepted':False,'gt_available':False})
if __name__=='__main__':
 try:main()
 except Exception:
  write('FAILURE.json',{'error':traceback.format_exc()});raise

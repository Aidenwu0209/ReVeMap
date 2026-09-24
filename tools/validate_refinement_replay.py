"""Recompute grounding and refined labels from fixed inputs and naming decisions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    protected={str(p):sha(p) for p in args.reference.rglob('*') if p.is_file()}
    for arm in ('baseline','candidate'):
        root=args.output/arm;root.mkdir()
        for name in ('runtime.json','DECISIONS.json','INPUT_PLAN.json'):
            shutil.copyfile(args.reference/name,root/name)
        for name in ('inputs','objects'):
            (root/name).symlink_to((args.reference/name).resolve(),target_is_directory=True)
        env={**os.environ,'PYTHONPATH':plan[arm+'_source'],'OMP_NUM_THREADS':'2',
             'OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        for stage,python in [('ground',plan['sam3_python']),('apply',plan['cpu_python'])]:
            with (root/(stage+'.log')).open('x') as log:
                subprocess.run([python,'-m','pose_pipeline.semantic_runtime.refinement',
                    '--workspace',str(root),'--stage',stage],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    import numpy as np
    a=args.output/'baseline';b=args.output/'candidate'
    arrays=0;errors=[]
    for part in ('grounding','refined'):
        for f in sorted((a/part).rglob('*.npz')):
            relative=f.relative_to(a)
            with np.load(f) as x,np.load(b/relative) as y:
                for key in x.files:
                    arrays+=1
                    if x[key].dtype!=y[key].dtype or not np.array_equal(x[key],y[key]):errors.append(str(relative)+':'+key)
    maps={}
    for f in (a/'refined').rglob('semantic_labeled.ply'):
        relative=f.relative_to(a)
        maps[str(relative)]={'baseline':sha(f),'candidate':sha(b/relative)}
        if sha(f)!=sha(b/relative):errors.append(str(relative))
    unchanged=all(sha(p)==digest for p,digest in protected.items())
    result={'scope':'fresh grounding and apply; fixed geometry, objects and VLM decisions; no fresh VLM',
            'arrays_checked':arrays,'maps':maps,'errors':errors,'source_unchanged':unchanged,
            'passed':not errors and unchanged}
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()

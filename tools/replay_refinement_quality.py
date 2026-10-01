"""Run the live refinement bridge against a sealed cached fusion candidate."""
import argparse
from pathlib import Path
import shutil

from pose_pipeline.live_semantic import prepare_refinement
from pose_pipeline.semantic_runtime.common import read,write,sha
from pose_pipeline.semantic_runtime.refinement.__main__ import run


def replay(reference,candidate,output):
    reference=reference.resolve(strict=True);candidate=candidate.resolve(strict=True)
    lock=read(candidate.parent/'PREDICTIONS_LOCK.json')
    if any(sha(candidate.parent/p)!=h for p,h in lock.items()):
        raise ValueError('candidate prediction lock changed')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'mapping').mkdir();(output/'fused').mkdir()
    source=reference/'mapping/mapping_result.json'
    shutil.copyfile(source,output/'mapping/mapping_result.json')
    for path,name in [(candidate/'map_labels.npz','map_labels.npz'),(candidate/'candidates.npz','candidates.npz'),
                      (reference/'fused/target.npz','target.npz'),(reference/'fused/classes.json','classes.json')]:
        shutil.copyfile(path,output/'fused'/name)
    write(output/'fused/CANDIDATES.json',{'labels_sha256':sha(output/'fused/map_labels.npz'),
        'candidates_sha256':sha(output/'fused/candidates.npz')})
    root=prepare_refinement(output,Path(read(source)['manifest']),reference/'runtime.json')
    run(root,'all')
    print(read(root/'refined/capture/RESULT.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    replay(args.reference,args.candidate,args.output)

"""Measure whether rejected points actually have multiple qualified input views.

This is a posthoc explanation of fixed loss reasons, not an inference change.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/src'))
from pose_pipeline.sam3_loss_audit import REASONS


def main():
    rows=[]
    for job in json.loads((ROOT/'inputs/JOBS.json').read_text()):
        path=ROOT/'loss_audit'/job['key']
        if not (path/'result.json').exists():
            raise ValueError('wait for complete audit batch')
        with np.load(path/'point_loss_reasons.npz') as z:
            reasons=z['reason'].copy(); lost=(z['old_instance']>0)&(z['matched_instance']==0)
        views=np.zeros(job['expected_points'],np.uint16)
        frame_hashes={}
        for fid in job['selected_frame_ids']:
            p=Path(job['cache_root'])/'frames'/f'{fid:06}.npz'
            frame_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(p) as z:
                ids=z['visible_map_ids']; masks=z['projected_local_instance']
                safe=z['interior']&(z['projected_semantic']>0)&(z['projected_confidence']>=.5)
                picked=np.zeros(len(ids),bool)
                for mask in np.unique(masks[safe]):
                    if mask<=0:continue
                    selected=safe&(masks==mask)
                    if selected.sum()>=30:picked|=selected
                # One point one vote per distinct original frame, independent of prompt count.
                views[ids[picked]]+=1
        breakdown={}
        for code,name in enumerate(REASONS):
            selected=lost&(reasons==code); values=views[selected]
            if not len(values):continue
            breakdown[name]={'points':len(values),'points_with_at_least_two_input_views':int(np.sum(values>=2)),
                'points_with_at_least_three_input_views':int(np.sum(values>=3)),
                'view_count_min':int(values.min()),'view_count_median':float(np.median(values)),
                'view_count_max':int(values.max())}
        report={'key':job['key'],'definition':'distinct original frames with interior, nonzero semantic, confidence>=0.5, mask size>=30 original map points; counted before graph filtering',
            'result_is_explanation_not_threshold_selection':True,'lost_reason_view_support':breakdown,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'frame_sha256':frame_hashes}
        (path/'loss_view_support.json').write_text(json.dumps(report,indent=2)+'\n')
        rows.append({'key':job['key'],'loss_views':breakdown})
    (ROOT/'LOSS_VIEW_SUPPORT.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

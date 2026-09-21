"""Frozen one-frame assisted blue-equipment diagnostic; no box/prompt tuning."""
from pathlib import Path
import json, time, traceback
import run_tracking_probe as probe


PLAN={'version':'blue-equipment-supplement-v1','scene':'orbbec/scan_20260909_142829_5ef1fa',
      'frame_id':4811,'rgb_size':[640,480], 'positive_box_xyxy':[169,0,431,335],
      'negative_box_xyxy':[440,165,620,300],
      'arms':['text_machine','text_blue_machine','visual_positive','visual_positive_negative'],
      'text_prompts':{'text_machine':'machine','text_blue_machine':'blue machine'},
      'selection':'raw RGB inspected by parent; prompt strings and boxes fixed before supplementary inference',
      'assisted_extra_supervision':True,'semantic_business_name_known':False,
      'gt_consumed':False,'quality_accepted':False,'no_iterations_or_prompt_search':True}


def box_normalized(xyxy):
    x1,y1,x2,y2=xyxy;w,h=PLAN['rgb_size']
    return [(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h]


def main():
    O=probe.R/'tracking/blue_equipment';O.mkdir(parents=True,exist_ok=False)
    probe.write_json(O/'PLAN.json',PLAN);started=time.monotonic()
    try:
        torch=probe.setup();before=probe.code_provenance()
        frames,audit=probe.load_clip({'key':PLAN['scene'],'frame_ids':[4811]});f=frames[0]
        assert list(f['image'].size)==PLAN['rgb_size'];probe.write_json(O/'INPUTS.json',audit)
        before['supplement_runner_sha256']=probe.sha256_file(Path(__file__))
        probe.write_json(O/'PROVENANCE_BEFORE.json',before)
        processor,model=probe.load_model(probe.A/'sam3.pt',probe.CHECKPOINT_SHA)
        probe.write_json(O/'MODEL.json',model);records=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_image(f['image'])
            for arm in PLAN['arms']:
                processor.reset_all_prompts(state);t=time.monotonic()
                if arm in PLAN['text_prompts']:
                    out=processor.set_text_prompt(PLAN['text_prompts'][arm],state)
                else:
                    out=processor.add_geometric_prompt(box_normalized(PLAN['positive_box_xyxy']),True,state)
                    if arm=='visual_positive_negative':
                        out=processor.add_geometric_prompt(box_normalized(PLAN['negative_box_xyxy']),False,state)
                row=probe.save_frame(O/arm,f,out,'image');masks,ids,scores=probe.normalize_output(out,'image')
                row=probe.compact([row])[0];row.update(arm=arm,seconds=time.monotonic()-t,scores=scores.tolist())
                probe.overlay(f['image'],masks,ids,arm).save(O/arm/'overlay.png');records.append(row)
        f['image'].save(O/'rgb.png')
        # Verify the actual raw RGB-D, map and poses after all four arms.
        check={**audit['inputs'],f['color_path']:f['color_sha256'],f['depth_path']:f['depth_sha256']}
        assert all(probe.sha256_file(Path(p))==h for p,h in check.items())
        after=probe.code_provenance();after['supplement_runner_sha256']=probe.sha256_file(Path(__file__))
        assert before==after;probe.write_json(O/'PROVENANCE_AFTER.json',after)
        status={'status':'completed','seconds':time.monotonic()-started,'arms':records,
                'same_inputs_all_arms':True,'all_input_hashes_unchanged':True,'provenance_unchanged':True,
                'accuracy_measured':False,'semantic_name_inferred':False}
    except BaseException:
        status={'status':'failed','error':traceback.format_exc()};traceback.print_exc()
    probe.write_json(O/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()

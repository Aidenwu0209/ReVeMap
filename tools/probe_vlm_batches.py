"""Offline batch-size probe for Qwen3-VL; never changes production naming."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=96)
    args = p.parse_args()
    from pose_pipeline.semantic_runtime.common import read, write, sha, PROMPT, parse_label
    from pose_pipeline.semantic_runtime.vlm import create_namer
    import numpy as np
    import torch
    from PIL import Image
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = read(args.tasks)
    crops = [c for t in tasks for c in t['crops']]
    indices = np.linspace(0, len(crops)-1, min(args.count,len(crops)), dtype=int)
    selected = [crops[i] for i in indices]
    if any(sha(c['file']) != c['sha256'] for c in selected):
        raise ValueError('crop digest changed')
    write(args.output/'INPUTS.json', {'selection':'uniform ordered crop indices',
                                    'indices':indices, 'crops':selected})
    config=read(args.runtime)
    model=create_namer('qwen3vl_2b_nf4', config['models']['qwen3vl_2b_nf4'])
    write(args.output/'MODEL.json', model.audit)
    # Warm up before the paired steady-state comparisons; same fixed model.
    model.infer(selected[0]['file'])
    results=[]
    try:
        for batch_size in (1, 2, 4, 1):
            torch.cuda.reset_peak_memory_stats()
            tick=time.perf_counter();answers=[]
            if batch_size == 1:
                answers=[model.infer(c['file']) for c in selected]
            else:
                old_padding=model.processor.tokenizer.padding_side
                model.processor.tokenizer.padding_side='left'
                try:
                    for start in range(0,len(selected),batch_size):
                        messages=[]
                        for crop in selected[start:start+batch_size]:
                            with Image.open(crop['file']) as im:
                                rgb=im.convert('RGB')
                            messages.append([{'role':'user','content':[{'type':'image','image':rgb},
                                {'type':'text','text':PROMPT}]}])
                        inputs=model.processor.apply_chat_template(messages,tokenize=True,
                            add_generation_prompt=True,return_dict=True,return_tensors='pt',
                            padding=True,enable_thinking=False).to('cuda')
                        for key,value in inputs.items():
                            if isinstance(value,torch.Tensor) and value.is_floating_point():
                                inputs[key]=value.to(model.dtype)
                        length=inputs['input_ids'].shape[-1]
                        with torch.inference_mode():
                            output=model.model.generate(**inputs,do_sample=False,
                                max_new_tokens=model.spec['max_new_tokens'],use_cache=True)
                        torch.cuda.synchronize()
                        for answer in model.processor.batch_decode(output[:,length:],skip_special_tokens=True):
                            label,valid=parse_label(answer)
                            answers.append({'raw_response':answer,'label':label,'valid':valid})
                finally:
                    model.processor.tokenizer.padding_side=old_padding
            torch.cuda.synchronize()
            entry={'batch_size':batch_size,'seconds':time.perf_counter()-tick,'answers':answers,
                   'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
            results.append(entry)
            write(args.output/'RESULTS.json', results)
            print(batch_size,entry['seconds'],flush=True)
    finally:
        model.close()
    base=results[0]['answers']
    for row in results:
        row['label_differences']=[{'index':i,'frame_id':selected[i]['frame_id'],
                                  'mask_id':selected[i]['mask_id'],'baseline':a['label'],'candidate':b['label']}
                                 for i,(a,b) in enumerate(zip(base,row['answers'])) if a['label']!=b['label']]
        row['raw_response_differences']=sum(a['raw_response']!=b['raw_response'] for a,b in zip(base,row['answers']))
    write(args.output/'RESULTS.json', results)
    write(args.output/'COMPLETE.json', {'scope':'96-crop prototype only; not enabled in production',
        'crops':len(selected),'batch_sizes':[1,2,4,1],
        'serial_repeat_equal':not results[-1]['raw_response_differences'],
        'all_inputs_unchanged':all(sha(c['file'])==c['sha256'] for c in selected)})


if __name__=='__main__':
    main()

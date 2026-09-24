"""Profile the existing VLM worker using saved crop tasks; never change generation."""
import argparse
from collections import Counter, defaultdict
import cProfile
import json
from pathlib import Path
import resource
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='qwen3vl_2b_nf4')
    p.add_argument('--reference', type=Path, help='Optional original VLM output for label comparison')
    args = p.parse_args()
    from pose_pipeline.semantic_runtime import vlm, worker
    from pose_pipeline.semantic_runtime.common import read, sha, write
    import torch
    config, tasks = read(args.runtime), read(args.tasks)
    if vlm.model_spec(args.model)['kind'] != 'native':
        raise ValueError('This profiler requires an explicit local Transformers model')
    metrics = defaultdict(lambda: {'seconds': 0., 'calls': 0})

    def wrap(label, function, *, sync=False):
        def timed(*a, **kw):
            if sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = function(*a, **kw)
            if sync:
                torch.cuda.synchronize()
            metrics[label]['seconds'] += time.perf_counter() - start
            metrics[label]['calls'] += 1
            return result
        return timed

    old_create, old_verify = vlm.create_namer, vlm.verify_weights
    vlm.verify_weights = wrap('weight_verification', old_verify)

    def create(*a, **kw):
        model = wrap('create_namer_including_import_verify_load', old_create)(*a, **kw)
        model.processor.apply_chat_template = wrap('image_text_preprocessing', model.processor.apply_chat_template)
        model.processor.batch_decode = wrap('text_decoding', model.processor.batch_decode)
        model.model.generate = wrap('generate_including_vision_prefill_decode', model.model.generate, sync=True)
        return model

    vlm.create_namer = create
    profiler = cProfile.Profile()
    start = time.perf_counter()
    try:
        profiler.enable()
        worker.name_tasks(tasks, args.model, config, args.output)
        profiler.disable()
    finally:
        vlm.create_namer, vlm.verify_weights = old_create, old_verify
    wall = time.perf_counter() - start
    profiler.dump_stats(str(args.output / 'profile.pstats'))
    import io, pstats
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(70)
    (args.output / 'profile.txt').write_text(stream.getvalue())
    records = read(args.output / 'RECORDS.json')
    durations = sorted(r['request_seconds'] for r in records)
    tokens = sum(r.get('output_tokens') or 0 for r in records)
    digests = Counter(c['sha256'] for t in tasks for c in t['crops'])
    report = {'scope': 'instrumented original local VLM; timings include cProfile and synchronization overhead',
              'wall_seconds': wall, 'components': dict(metrics),
              'component_nesting': 'weight_verification is inside create_namer; preprocessing/generate/decoding are inside requests',
              'requests_sum_seconds': sum(durations), 'crop_count': len(records), 'output_tokens': tokens,
              'median_request_seconds': durations[len(durations)//2] if durations else None,
              'p95_request_seconds': durations[min(len(durations)-1, int(.95*len(durations)))] if durations else None,
              'identical_crop_extra_requests': sum(v-1 for v in digests.values()),
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024),
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'tasks_sha256': sha(args.tasks), 'model': read(args.output / 'MODEL.json'),
              'source_sha256': {str(Path(m.__file__)): sha(m.__file__) for m in (vlm, worker)}}
    if args.reference:
        previous = {(r['frame_id'],r['mask_id']):r for r in read(args.reference/'RECORDS.json')}
        differences = []
        for row in records:
            key = (row['frame_id'],row['mask_id'])
            if key not in previous or any(row[k] != previous[key][k] for k in ('label','raw_response','valid','sha256')):
                differences.append(list(key))
        report['reference_comparison'] = {'reference': str(args.reference), 'differences': differences,
                                         'same_count': len(records)==len(previous)}
    write(args.output / 'PROFILE_SUMMARY.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

"""Benchmark one SAM3 stage on real RGB-D input, without running mapping/VLM.

Run with the target checkout's src on PYTHONPATH and its SAM3 interpreter.
Outputs are always new; profiling is opt-in to separate diagnosis from timing.
"""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    from pose_pipeline.semantic_runtime.common import read, sha, write
    from pose_pipeline.semantic_runtime.worker import infer_sam3
    import torch
    config = read(args.runtime)
    profiler = cProfile.Profile() if args.profile else None
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    infer_sam3(args, config)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.output / 'profile.pstats'))
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(70)
        (args.output / 'profile.txt').write_text(stream.getvalue())
    from pose_pipeline import sam3_refine, sam3_fusion, sam3_mapping
    from pose_pipeline.semantic_runtime import worker
    sources = [Path(module.__file__) for module in (worker, sam3_refine, sam3_fusion, sam3_mapping)]
    cache_source = Path(sam3_mapping.__file__).with_name('sam3_text_cache.py')
    if cache_source.exists():
        sources.append(cache_source)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {'scope': 'SAM3 stage only; model load, inference and exports included',
              'seconds': seconds, 'cpu_seconds': usage.ru_utime + usage.ru_stime,
              'peak_rss_mib': usage.ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024),
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
              'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
              'profile_enabled': bool(profiler), 'manifest': str(args.manifest),
              'manifest_sha256': sha(args.manifest),
              'source_sha256': {str(p): sha(p) for p in sources},
              'stage': read(args.output / 'COMPLETE.json')}
    write(args.output / 'PROFILE_SUMMARY.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

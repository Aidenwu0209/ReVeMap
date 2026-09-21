# SAM3 cached multi-frame instance fusion experiment

Status: opt-in research candidate only. Keep `08f8754` C (`objects_geometry`) as the general baseline; do not promote to `developnew`.

`pose_pipeline.sam3_multiview.fuse_instances` consumes per-frame depth-consistent projected mask IDs on an immutable map. It filters repeated split masks, uses sparse-view agreement with unknown abstention, and applies component-level cannot-link constraints. Existing semantic labels are fixed; this does not add custom object recognition or predict new graph relationships.

Two shared profiles were run on ScanNet 0030/0050/0011, Orbbec 4812 and six archived 3RScan frames. The strict profile changed output gates relative to C. The matched profile restored mask admission to 30 points, allowed single-view points in multi-frame objects, and used component per-point maximum confidence, while keeping the same graph principles. See PLAN_MATCHED.json for the remaining differences from C. It was fixed before reading new GT evaluations; no scene-specific threshold search was performed.

Matched ScanNet0030: GT instance coverage 53.739% to 56.134%, purity 93.212% to 94.271%, mixed instances 22 to 11, IoU25 object recall 17/27 to 22/27, IoU50 stays 11/27. Orbbec map instance coverage falls 21.332% to 13.870%. All semantic and confidence arrays and original map geometry are unchanged. These fixed 27-object diagnostics are not official instance AP; 0030 is not a blind holdout. Orbbec has no GT here, and six 3RScan frames are not full-scene evidence.

59 targeted tests passed: 45 new multiview contracts plus 14 existing SAM3 tests. Ten maps passed input hashes, original vertex preservation, object-inventory and visual-layout audits. The previous sealed 426 files are unchanged. No environment install or new model inference was performed.

The frozen runner beside this document expects to be copied into a new result root containing `code/src`, `inputs/JOBS.json`, `ENV_REUSE.json`, PLAN.json and PLAN_MATCHED.json. It refuses existing output scenes. The original experiment root contains the complete input manifest and evaluation/render utilities. Execute with the existing ssh44 R1 environment:

```sh
/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python \
  /path/to/new/result/root/run_multiview.py --stage consensus_matched
```

Artifacts and Chinese report:
`/Users/wu/Desktop/wu/SGF-SGA/comparisons/sam3_multiview_20260912_v1/REPORT_zh.md`

Remote artifacts:
`/mnt/d/SGF-SGA-experiments/sam3_multiview_20260912_v1`

Method inspiration: https://arxiv.org/html/2401.07745v2. This adaptation does not implement full official MaskClustering or its appearance-feature aggregation. It is not wired into the production pipeline by default. New scene_graph relations are intentionally empty because old endpoints refer to different instance IDs.

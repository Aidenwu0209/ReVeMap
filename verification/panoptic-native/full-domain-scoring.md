# Replaying the full-domain ScanNet evaluation

Run this command on one completed map from the repository root. Supply the original estimated trajectory used to define the map coordinate frame and the official ScanNet scene directory and label table. Python dependencies are NumPy, SciPy and plyfile.

```sh
python tools/score_scannet_full_gt.py \
  --scan "$SCANNET_ROOT/scene0011_00" \
  --label-tsv "$SCANNET_LABEL_TSV" \
  --trajectory "$ORIGINAL_TRAJECTORY_JSON" \
  --prediction "$FINAL_MAP_PLY" --classes "$CLASSES_JSON" > scene0011_00.json
```

For canonical NPZ predictions, pass the NPZ to `--prediction` and omit `--classes`. Its arrays are `xyz` (N × 3 metres), `semantic` (ScanNet20 zero-based in the order in `tools/panoptic_observations.py`, 20 for unknown), and `instance` (positive thing IDs). PLY input uses `x`, `y`, `z`, `semantic_id` and `instance_id`. The classes JSON maps stringified PLY label IDs to ScanNet20 names.

The evaluator anchors all arms at the first common valid reference and original estimated camera pose. It projects the nearest map point onto each reference vertex within 5 cm. Other vertices receive the unknown class and void instance. Within each scan, PQ and mIoU average the classes present in its reference. Compare the unweighted scan mean to the accompanying 142-row CSV. The matched reference IDs support paired counting of recovered and lost objects. The output also gives coverage on labeled reference vertices, precision on all map points and the bidirectional mean nearest-neighbor distance on full-resolution coordinates.

The release wrapper was checked against the independent audit on all three sealed maps of scene0011_00. PQ and mIoU agreed within 1e-10 percentage points, thing TP/FP/FN and the anchor frame agreed exactly, and both geometric percentages and mean distance agreed within 1e-10. The 142-scan evidence comes from the separately hashed audit program reported in the audit JSON.

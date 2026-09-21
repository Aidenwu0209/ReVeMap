# Conservative guide-assisted recovery experiment

The final CPU arm is `guided_recovery_maskveto`, invoked by
`run_guided_backend_maskveto.py`. It uses frozen C object IDs as tentative 3D
guides and adds ownership only to C-owned / R3-matched-unassigned points.
It preserves every existing R3 owner and all non-instance fields.

This is a bounded MV3DIS-inspired common-guide adaptation, not a reproduction
of its 3D mask selector, superpoint pipeline, or continuous depth weighting.
A guide is a hypothesis, not correctness evidence. New point ownership still
requires direct depth-checked mask interior observations in two distinct
frames and maximum confidence at least 0.8. Observed repeated splits and
multiple significant R3 anchors veto guide recovery. Ambiguous shared masks
cannot provide positive support in that frame; other unambiguous frames may
still support the same guide. Mask
nodes already filtered as undersegmented by R3 cannot supply positive support.
No unobserved point, semantic-unknown point, or point outside a C guide is filled.

The original `guided_recovery` arm and its source bytes are preserved under
R4 `guided_code/` and `run_guided_backend.py`. This preliminary arm lacked the
explicit prior undersegmentation-mask veto and is superseded for evaluation.
The corrective arm was frozen before this agent read any new GT, with the
same thresholds and five-scene selection. See `PLAN_GUIDED_MASKVETO.json`.

Nine local safety/causal tests passed. Both CPU arms completed all five scenes.
Final output includes `map_labels.npz`, `objects.json`, `classes.json`,
`fusion_audit.json`, exact `perpoint_provenance.npz`, per-scene input/source
SHA-256 hashes, and independent support verification. Scene graph relations
are empty because this arm predicts no relations and does not remap old edges.
The parent task owns GT quality evaluation, map rendering, and adoption.

Final result root:
`/Users/wu/Desktop/wu/SGF-SGA/comparisons/sam3_guided_20260912_v1/guided_recovery_maskveto`
Remote root:
`/mnt/d/SGF-SGA-experiments/sam3_guided_20260912_v1/guided_recovery_maskveto`
Runtime:
`/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python`

3RScan is still a six-cached-frame diagnostic, not a full-scene run. Instance
coverage gains do not establish segmentation accuracy or semantic correctness.

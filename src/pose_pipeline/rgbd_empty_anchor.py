"""Keep unused empty anchors without inventing measured depth constraints.

This is the original topology-based native refill policy. Empty anchors are
allowed only outside every non-keyframe source set. The actual, masked BA
inputs must retain finite nonzero measured-source support for every target;
this necessary support check does not certify accuracy or observability.
"""

from __future__ import annotations

import bisect
import json
from numbers import Integral
from pathlib import Path


def audit_observations(
    *,
    ii,
    jj,
    target_start,
    target_end,
    source_used_nodes,
    empty_source_nodes,
    source_valid_pixels,
    total_target_components,
    finite_target_components,
    total_weight_components,
    finite_weight_components,
    nonzero_weight_components,
    nonzero_weight_pixels,
):
    if (
        not isinstance(target_start, Integral)
        or not isinstance(target_end, Integral)
        or not 0 <= target_start < target_end
    ):
        raise ValueError("nonempty integer BA target range required")
    arrays = [
        ii,
        jj,
        source_valid_pixels,
        total_target_components,
        finite_target_components,
        total_weight_components,
        finite_weight_components,
        nonzero_weight_components,
        nonzero_weight_pixels,
    ]
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("one statistic per actual edge required")
    if not all(
        isinstance(v, Integral) and not isinstance(v, bool) for a in arrays for v in a
    ):
        raise ValueError("exact integral counts/node IDs required")
    used = set(source_used_nodes)
    empty = set(empty_source_nodes)
    if not all(
        isinstance(v, Integral) and not isinstance(v, bool) and 0 <= v < target_start
        for v in used | empty
    ):
        raise ValueError("source/empty node IDs must be fixed anchor indices")
    failures = []
    if used & empty:
        failures.append(
            dict(
                reason="empty_anchor_in_frozen_NF_source_set",
                nodes=sorted(used & empty),
            )
        )
    targets = {
        t: dict(
            node=t,
            edge_count=0,
            measured_source_nodes=[],
            nonzero_weight_components=0,
            nonzero_weight_pixels=0,
            all_target_components_finite=True,
            all_masked_weight_components_finite=True,
        )
        for t in range(target_start, target_end)
    }
    for e, values in enumerate(zip(*arrays)):
        i, j, source_n, total_t, finite_t, total_w, finite_w, nonzero_w, nonzero_px = (
            map(int, values)
        )
        row = dict(edge=e, source=i, target=j)
        if i not in used or not 0 <= i < target_start:
            failures.append(dict(**row, reason="source_outside_frozen_NF_source_set"))
        if i in empty or source_n == 0:
            failures.append(dict(**row, reason="actual_edge_has_empty_measured_source"))
        if j not in targets:
            failures.append(dict(**row, reason="target_outside_BA_NF_range"))
            continue
        valid_counts = (
            source_n >= 0
            and total_t > 0
            and total_w > 0
            and total_t == total_w
            and total_w % 2 == 0
            and 0 <= finite_t <= total_t
            and 0 <= finite_w <= total_w
            and 0 <= nonzero_w <= total_w
            and 0 <= nonzero_px <= total_w // 2
            and nonzero_px <= nonzero_w <= 2 * nonzero_px
            and nonzero_px <= source_n <= total_w // 2
        )
        if not valid_counts:
            failures.append(dict(**row, reason="inconsistent_tensor_reduction_counts"))
        target = targets[j]
        target["edge_count"] += 1
        if source_n > 0:
            target["measured_source_nodes"].append(i)
        target["nonzero_weight_components"] += nonzero_w
        target["nonzero_weight_pixels"] += nonzero_px
        if finite_t != total_t:
            target["all_target_components_finite"] = False
            failures.append(dict(**row, reason="nonfinite_target_coordinates"))
        if finite_w != total_w:
            target["all_masked_weight_components_finite"] = False
            failures.append(dict(**row, reason="nonfinite_masked_weights"))
    for target in targets.values():
        target["measured_source_nodes"] = sorted(set(target["measured_source_nodes"]))
        if not target["edge_count"]:
            failures.append(
                dict(target=target["node"], reason="NF_target_has_no_actual_edges")
            )
        elif target["nonzero_weight_components"] == 0:
            failures.append(
                dict(
                    target=target["node"],
                    reason="NF_target_has_zero_measured_visual_weight",
                )
            )
    return dict(
        ok=not failures,
        failures=failures,
        targets=list(targets.values()),
        actual_edge_count=len(ii),
        target_count=target_end - target_start,
        unused_empty_anchor_nodes=sorted(empty - used),
        changes_weights=False,
        numerical_weight_threshold=0,
        scope="Necessary finite/nonzero measured-visual support only; does not certify observability rank or pose quality",
    )


def source_topology(anchor_ordinals, raw_frame_count):
    ids = [int(x) for x in anchor_ordinals]
    n = int(raw_frame_count)
    if (
        not ids
        or ids[0] != 0
        or ids[-1] != n - 1
        or any(a >= b for a, b in zip(ids, ids[1:]))
    ):
        raise ValueError("full strictly ordered anchor endpoints required")
    known = set(ids)
    nf = [t for t in range(n) if t not in known]
    rows = []
    used = set()
    for order, t in enumerate(nf):
        i = bisect.bisect_right(ids, t) - 1
        j = i + 1 if i < len(ids) - 1 else i
        used.update([i, j])
        rows.append(
            {
                "raw_ordinal": t,
                "source_nodes": [i, j],
                "source_raw_ordinals": [ids[i], ids[j]],
                "batch16": order // 16,
                "target_local_index": order % 16,
            }
        )
    return {
        "anchor_ordinals": ids,
        "raw_frame_count": n,
        "nonkeyframes": rows,
        "source_used_nodes": sorted(used),
        "original_filler_selection_rule": "count(ts<=t)-1 then min(next,N-1)",
        "gt_consumed": False,
    }


def permit_unused_empty(anchor_node, source_used_nodes):
    if int(anchor_node) in set(source_used_nodes):
        raise RuntimeError(
            "An empty measured-depth anchor is used by an actual native nonkeyframe source"
        )
    return True


class Policy:
    def __init__(self, anchor_ordinals, raw_frame_count, output):
        self.topology = source_topology(anchor_ordinals, raw_frame_count)
        self.ids = self.topology["anchor_ordinals"]
        self.used = set(self.topology["source_used_nodes"])
        self.empty = set()
        self.output = Path(output)
        self.calls = 0
        self.targets_seen = set()
        self.log = None
        self.cursor = 0
        self.batch_number = 0
        self.active_raw_targets = []
        self.raw_targets_seen = set()
        self.expected_sources = {
            r["raw_ordinal"]: set(r["source_nodes"])
            for r in self.topology["nonkeyframes"]
        }
        with (self.output / "native_source_topology.json").open("x") as f:
            json.dump(self.topology, f, indent=2)

    def allow_unused_empty(self, node):
        permit_unused_empty(node, self.used)
        self.empty.add(int(node))

    def begin_batch(self, timestamps):
        actual = [int(t) for t in timestamps]
        expected = [
            r["raw_ordinal"]
            for r in self.topology["nonkeyframes"][
                self.cursor : self.cursor + len(actual)
            ]
        ]
        if not actual or len(actual) > 16 or actual != expected:
            raise RuntimeError("Native NF batch differs from frozen original order")
        self.active_raw_targets = actual
        self.cursor += len(actual)
        self.batch_number += 1

    def assert_empty_mono_zero(self, video):
        # Read only; never fill/assign any mono_disps or mono_disps_up values.
        for node in self.empty:
            if bool((video.mono_disps[node] != 0).any()):
                raise RuntimeError(
                    "Allowed unused empty anchor acquired measured mono disparity"
                )

    def audit_ba(
        self, video, target, masked_weight, source_valid, ii, jj, args, kwargs
    ):
        """Reduce actual incoming BA tensors; do not modify or replace them."""
        import torch

        self.assert_empty_mono_zero(video)
        t0 = int(args[0] if len(args) > 0 else kwargs["t0"])
        t1 = int(args[1] if len(args) > 1 else kwargs["t1"])
        if t0 != len(self.ids):
            raise RuntimeError("Native BA must keep all graph anchors fixed")
        if t1 - t0 != len(self.active_raw_targets):
            raise RuntimeError(
                "Actual BA target range differs from current raw NF batch"
            )
        src = [int(x) for x in ii.detach().cpu().tolist()]
        dst = [int(x) for x in jj.detach().cpu().tolist()]
        E = len(src)
        for i, j in zip(src, dst):
            if (
                not t0 <= j < t1
                or i not in self.expected_sources[self.active_raw_targets[j - t0]]
            ):
                raise RuntimeError(
                    "Actual BA edge differs from the original NF bracketing sources"
                )
        for j in range(t0, t1):
            actual = sorted(i for i, destination in zip(src, dst) if destination == j)
            expected = sorted(self.expected_sources[self.active_raw_targets[j - t0]])
            if actual != expected:
                raise RuntimeError(
                    "Actual native source multiset differs for raw NF "
                    + str(self.active_raw_targets[j - t0])
                    + ": "
                    + str(actual)
                    + " versus "
                    + str(expected)
                )
        if (
            target.shape != masked_weight.shape
            or target.shape[-1] != 2
            or target.shape[-4] != E
        ):
            raise RuntimeError("Unexpected original native target/weight tensor shape")
        tv = target.detach().movedim(-4, 0).reshape(E, -1)
        wv = masked_weight.detach().movedim(-4, 0).reshape(E, -1)
        if source_valid.shape[0] != E:
            raise RuntimeError("Measured source mask edge count differs")
        tf = torch.isfinite(tv).sum(dim=1).cpu().tolist()
        wf = torch.isfinite(wv).sum(dim=1).cpu().tolist()
        nz = torch.isfinite(wv) & (wv != 0)
        nc = nz.sum(dim=1).cpu().tolist()
        npix = nz.reshape(E, -1, 2).any(dim=-1).sum(dim=1).cpu().tolist()
        sv = source_valid.detach().reshape(E, -1).sum(dim=1).cpu().tolist()
        # All reductions read inputs only; the exact masked_weight tensor is
        # still passed to the unchanged native BA call by the original wrapper.
        edge_stats = [
            {
                "target_components": int(tv.shape[1]),
                "finite_target_components": int(tf[e]),
                "weight_components": int(wv.shape[1]),
                "finite_weight_components": int(wf[e]),
                "nonzero_weight_components": int(nc[e]),
                "nonzero_weight_pixels": int(npix[e]),
                "source_valid_pixels": int(sv[e]),
            }
            for e in range(E)
        ]
        audit = audit_observations(
            ii=src,
            jj=dst,
            target_start=t0,
            target_end=t1,
            source_used_nodes=sorted(self.used),
            empty_source_nodes=sorted(self.empty),
            source_valid_pixels=[x["source_valid_pixels"] for x in edge_stats],
            total_target_components=[x["target_components"] for x in edge_stats],
            finite_target_components=[
                x["finite_target_components"] for x in edge_stats
            ],
            total_weight_components=[x["weight_components"] for x in edge_stats],
            finite_weight_components=[
                x["finite_weight_components"] for x in edge_stats
            ],
            nonzero_weight_components=[
                x["nonzero_weight_components"] for x in edge_stats
            ],
            nonzero_weight_pixels=[x["nonzero_weight_pixels"] for x in edge_stats],
        )
        self.calls += 1
        self.targets_seen.update(dst)
        self.raw_targets_seen.update(self.active_raw_targets[j - t0] for j in dst)
        for x in audit["targets"]:
            x["raw_ordinal"] = self.active_raw_targets[x["node"] - t0]
        record = {
            "ba_call": self.calls,
            "batch_number": self.batch_number,
            "raw_NF_targets": list(self.active_raw_targets),
            "ii": src,
            "jj": dst,
            "source_raw_ordinals": [self.ids[i] for i in src],
            "target_raw_ordinals": [self.active_raw_targets[j - t0] for j in dst],
            "t0": t0,
            "t1": t1,
            "edge_stats": edge_stats,
            "target_support": audit,
            "empty_anchor_nodes": sorted(self.empty),
            "empty_anchor_raw_ordinals": [self.ids[i] for i in sorted(self.empty)],
            "all_empty_mono_still_zero": True,
            "weight_tensor_modified_by_audit": False,
            "support_scope": "Actual BA input after unchanged measured-source weight mask; not a rank, correspondence or accuracy certificate",
        }
        if self.log is None:
            self.log = (self.output / "native_ba_target_support.jsonl").open("x")
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()
        if not audit["ok"]:
            raise RuntimeError(
                "Actual native BA target lacks finite nonzero measured-source support: "
                + json.dumps(audit["failures"])
            )

    def finalize(self, video):
        self.assert_empty_mono_zero(video)
        expected = {r["raw_ordinal"] for r in self.topology["nonkeyframes"]}
        if self.cursor != len(expected) or self.raw_targets_seen != expected:
            raise RuntimeError(
                "Some actual raw nonkeyframes did not receive audited BA support"
            )
        if self.log is not None:
            self.log.close()
        with (self.output / "native_empty_anchor_policy_result.json").open("x") as f:
            json.dump(
                {
                    "policy": "allow zero-measured anchor only when absent from all original native NF source sets",
                    "source_used_nodes": sorted(self.used),
                    "empty_anchor_nodes": sorted(self.empty),
                    "empty_anchor_raw_ordinals": [
                        self.ids[i] for i in sorted(self.empty)
                    ],
                    "zero_sensor_preserved": True,
                    "all_graph_anchor_poses_fixed_by_original_wrapper": True,
                    "ba_calls_audited": self.calls,
                    "all_observed_BA_targets_have_finite_nonzero_masked_weight": True,
                    "raw_nonkeyframes_with_audited_BA_support": sorted(
                        self.raw_targets_seen
                    ),
                    "raw_nonkeyframe_count": len(expected),
                    "actual_ii_never_contains_empty_unused_anchor": True,
                    "gt_consumed": False,
                    "quality_accepted": False,
                },
                f,
                indent=2,
            )

"""Review ownership of newly measured cells using a bounded history of anchored masks."""

import numpy as np


def _different_view(features, a, b):
    delta = np.abs(features[a] - features[b])
    return (delta[..., :3].max(axis=-1) >= 0.10) | (delta[..., 3:].max(axis=-1) >= 0.16)


def _remember(bank, cells, mask, frames, features):
    # One representative for consecutive, near-identical camera positions.
    old = bank[cells]
    shift = (old[:, 0] == 0) | _different_view(features, frames[old[:, 0]], frames[mask])
    old[shift, 1:] = old[shift, :-1]
    old[:, 0] = mask
    bank[cells] = old


def review_ownership(voxel_ids, nodes, meta, features, distances, nearest, base_semantic,
                     base_instance, append, added_semantic, added_instance, verified):
    """Propose an existing owner from earlier views and optionally check withheld masks.

    Both arms use exactly the same proposals. Only measured, mask-labelled voxels
    count as witnesses. An absent or occluded observation supplies no negative vote.
    """
    positive = nodes > 0
    order = np.argsort(nodes[positive], kind="stable")
    masks = nodes[positive][order]
    cells = voxel_ids[positive][order]
    ids, starts = np.unique(masks, return_index=True)
    audit = {"eligible_masks": 0, "historical_masks": 0, "proposed_cells": 0,
             "witnessed_cells": 0, "changed_cells": 0}
    if not len(ids):
        return added_instance, audit
    mask_cells = dict(zip(ids, np.split(cells, starts[1:])))
    frames = np.zeros(len(meta), dtype=int)
    owners = np.zeros(len(meta), dtype=base_instance.dtype)
    by_class = {}
    # Bin complete camera poses into disjoint discovery and witness groups.
    # Mixing the integer coordinates avoids assigning a whole room to one arm.
    pose_bins = np.floor(features / [0.25, 0.25, 0.25, 0.30, 0.30, 0.30]).astype(np.int64)
    witness_frame = (pose_bins @ np.array([1, 3, 7, 11, 13, 17])) % 5 == 0
    for mask, group in mask_cells.items():
        frame, category = meta[mask]
        frames[mask] = frame
        by_class.setdefault(category, []).append(mask)
        nearby = (distances[group] <= 0.05) & (base_semantic[nearest[group]] == category)
        votes = base_instance[nearest[group[nearby]]]
        values, counts = np.unique(votes[votes > 0], return_counts=True)
        if len(values):
            best = counts.argmax()
            if counts[best] >= max(15, 0.60 * counts.sum(), 0.15 * len(group)):
                owners[mask] = values[best]

    n = len(append)
    proposed = np.zeros(n, dtype=base_instance.dtype)
    ambiguous = np.zeros(n, bool)
    witnessed = np.zeros(n, bool)
    contradicted = np.zeros(n, bool)
    categories = np.zeros(n, dtype=added_semantic.dtype)
    categories[append] = added_semantic
    for category, class_masks in by_class.items():
        # Process one class at a time. Each voxel stores at most three masks per bank.
        previous = np.zeros((n, 3), dtype=np.int32)
        witnesses = np.zeros_like(previous) if verified else None
        if verified:
            for mask in class_masks:
                if owners[mask] and witness_frame[frames[mask]]:
                    _remember(witnesses, mask_cells[mask], mask, frames, features)
        for mask in class_masks:
            frame, group = frames[mask], mask_cells[mask]
            if witness_frame[frame]:
                continue
            if owners[mask]:
                _remember(previous, group, mask, frames, features)
                continue
            audit["eligible_masks"] += 1
            older, hits = np.unique(previous[group], return_counts=True)
            keep = (older > 0) & (hits >= 12) & _different_view(features, frames[older], frame)
            older, hits = older[keep], hits[keep]
            if not len(older):
                continue
            fractions = hits / np.array([min(len(group), len(mask_cells[k])) for k in older])
            older, hits = older[fractions >= 0.30], hits[fractions >= 0.30]
            if not len(older):
                continue
            values, index = np.unique(owners[older], return_inverse=True)
            support = np.bincount(index, weights=hits)
            best = support.argmax()
            if support[best] < 0.70 * support.sum():
                continue
            winner = values[best]
            # ID ties and contradictory witnesses retain the frozen baseline.
            chosen = group[append[group] & (categories[group] == category)]
            if not len(chosen):
                continue
            audit["historical_masks"] += 1
            ambiguous[chosen] |= (proposed[chosen] > 0) & (proposed[chosen] != winner)
            proposed[chosen] = winner
            if verified:
                source = older[np.argmax(np.where(owners[older] == winner, hits, 0))]
                w = witnesses[chosen]
                independent = ((w > 0) & _different_view(features, frames[w], frame)
                               & _different_view(features, frames[w], frames[source]))
                witnessed[chosen] |= np.any(independent & (owners[w] == winner), axis=1)
                contradicted[chosen] |= np.any(independent & (owners[w] != winner), axis=1)
    eligible = (proposed > 0) & ~ambiguous
    audit["proposed_cells"] = int(eligible.sum())
    if verified:
        eligible &= witnessed & ~contradicted
        audit["witnessed_cells"] = int(eligible.sum())
    changed = eligible[append] & (added_instance != proposed[append])
    output = added_instance.copy()
    output[changed] = proposed[append][changed]
    audit["changed_cells"] = int(changed.sum())
    return output, audit

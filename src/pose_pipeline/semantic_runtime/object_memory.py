"""Experimental object-level naming evidence; never changes map ownership.

All associations use predicted map point IDs. A reused name is metadata, not a
new observation: only actually inferred, distinct source frames may vote.
"""
from collections import Counter, defaultdict
import math


def crop_key(crop):
    return f'{int(crop["frame_id"]):06}_{int(crop["mask_id"]):04}'


def association(instance_ids):
    counts = Counter(int(x) for x in instance_ids if x > 0)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    total = len(instance_ids)
    if not ranked:
        return {"object_id": 0, "eligible": False, "share": 0., "candidates": []}
    oid, count = ranked[0]
    candidates = [{"object_id": i, "points": n, "share": n / max(total, 1)} for i, n in ranked]
    share = count / max(total, 1)
    return {"object_id": oid, "eligible": count >= 30 and share >= .65,
            "share": share, "visible_points": total, "candidates": candidates,
            "ambiguous": share < .8 or any(x["points"] >= 30 and x["share"] >= .15 for x in candidates[1:])}


def representatives(crops, limit=3):
    """Keep one crop per source frame, favour large, pure and diverse views."""
    frames = {}
    quality = lambda x: float(x["mask_pixels"]) * float(x["association"]["share"])
    for crop in crops:
        fid = crop["frame_id"]
        if fid not in frames or (quality(crop), crop_key(crop)) > (quality(frames[fid]), crop_key(frames[fid])):
            frames[fid] = crop
    remaining = sorted(frames.values(), key=lambda x: (-quality(x), crop_key(x)))
    chosen = []
    while remaining and len(chosen) < limit:
        best = quality(remaining[0]) if not chosen else max(quality(x) for x in remaining)
        pool = [x for x in remaining if quality(x) >= .5 * best]
        def score(x):
            if not chosen:
                return (quality(x), crop_key(x))
            direction = x.get("view_direction", [0., 0., 0.])
            novelty = min(1. - max(-1., min(1., sum(a*b for a,b in zip(direction, y.get("view_direction", [0.,0.,0.]))))) for y in chosen)
            return (novelty, quality(x), crop_key(x))
        selected = max(pool, key=score)
        chosen.append(selected)
        remaining.remove(selected)
    return chosen


def voted_name(records):
    votes = defaultdict(set)
    for crop in records:
        if crop.get("executed") and crop.get("label", "unknown") != "unknown":
            votes[crop["label"]].add(crop["frame_id"])
    ranked = sorted(votes, key=lambda k: (-len(votes[k]), k))
    if ranked and len(votes[ranked[0]]) >= 2 and (len(ranked) == 1 or len(votes[ranked[0]]) > len(votes[ranked[1]])):
        return ranked[0]
    return "unknown"


def cache_identity(scene, geometry_sha, instance_sha, model_revision, prompt_sha, crop_hashes):
    """A namespace changes with geometry, membership, model, prompt or evidence."""
    import hashlib, json
    fields = [scene, geometry_sha, instance_sha, model_revision, prompt_sha, sorted(crop_hashes)]
    return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()


def clip_accept(own_similarities, rival_similarities, min_similarity=.70, margin=.05):
    # No prototype means no evidence to alter the geometric association.
    if not own_similarities:
        return True, "no_independent_prototype_keep_geometry"
    if not all(math.isfinite(x) and -1.001 <= x <= 1.001 for x in [*own_similarities, *rival_similarities]):
        raise ValueError("invalid cosine similarity")
    own = max(own_similarities)
    if own < min_similarity:
        return False, "appearance_disagreement"
    if rival_similarities and own - max(rival_similarities) < margin:
        return False, "appearance_ambiguous"
    return True, "appearance_supported"

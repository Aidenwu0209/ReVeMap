"""Use native tracked objects to recover or extend a 3D model's instance masks."""
import numpy as np


CLASSES = ("wall floor cabinet bed chair sofa table door window bookshelf picture "
           "counter desk curtain refrigerator showercurtain toilet sink bathtub otherfurniture").split()
ALIASES = {"refridgerator":"refrigerator", "shower curtain":"showercurtain", "showercurtrain":"showercurtain"}


def reconcile(vertex, classes, semantic, instance, head, confidence, distance):
    native = np.full(len(vertex), 20, dtype=np.uint8)
    for key, name in classes.items():
        name = name.strip().lower()
        name = ALIASES.get(name, name)
        if name in CLASSES:
            native[vertex["semantic_id"] == int(key)] = CLASSES.index(name)
    native_id = vertex["instance_id"]
    semantic, instance = semantic.copy(), instance.copy()
    next_id = int(instance.max()) + 1
    tracked = (native_id > 0) & (native >= 2) & (native < 20)
    object_ids, sizes = np.unique(native_id[tracked], return_counts=True)
    for object_id in object_ids[sizes >= 100]:
        support = tracked & (native_id == object_id)
        category = np.bincount(native[support]).argmax()
        segment = support & (native == category)
        occupied = (semantic >= 2) & (instance > 0)
        agreement = segment & (head == category) & (confidence >= 0.5)
        if occupied[segment].mean() <= 0.30 and agreement.sum() >= 0.35 * segment.sum():
            take = segment & ~occupied
            semantic[take], instance[take] = category, next_id
            next_id += 1
            continue
        shared = segment & occupied & (semantic == category)
        if shared.sum() < 0.30 * segment.sum():
            continue
        ids, votes = np.unique(instance[shared], return_counts=True)
        best = votes.argmax()
        if votes[best] >= 0.80 * shared.sum():
            take = agreement & ~occupied & (distance <= 0.12)
            semantic[take], instance[take] = category, ids[best]
    return semantic, instance

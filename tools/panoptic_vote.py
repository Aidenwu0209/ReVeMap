"""Match native 3D instances across views and vote for their supported surfaces."""

import numpy as np


def majority_vote(predictions):
    # The tuple of panoptic labels across views partitions vertices into sets with
    # identical memberships. Weighted counts give exact intersections and unions.
    symbols = []
    for pred in predictions:
        classes, ids = pred["semantic"], pred["instance"]
        symbols.append(
            np.where(
                (classes >= 2) & (ids > 0),
                32 * ids.astype("i8") + classes,
                -1 - classes.astype("i8"),
            )
        )
    bins, restore, weight = np.unique(
        np.column_stack(symbols), return_inverse=True, return_counts=True, axis=0
    )
    restore = restore.reshape(-1)
    handles, classes, sizes, views = [], [], [], []
    for view, column in enumerate(bins.T):
        labels, inverse = np.unique(column, return_inverse=True)
        positive = labels >= 0
        count = positive.sum()
        ids = np.full(len(labels), -1)
        ids[positive] = np.arange(len(classes), len(classes) + count)
        handles.append(ids[inverse])
        classes.extend((labels[positive] % 32).tolist())
        sizes.extend(np.bincount(inverse, weights=weight)[positive].tolist())
        views.extend([view] * count)

    total = len(classes)
    parent = list(range(total))
    members = [{view} for view in views]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = []
    for a in range(len(predictions)):
        for b in range(a):
            x, y = handles[a], handles[b]
            overlap = (x >= 0) & (y >= 0)
            pairs, idx = np.unique(x[overlap] * total + y[overlap], return_inverse=True)
            intersections = np.bincount(idx, weights=weight[overlap])
            for pair, intersection in zip(pairs, intersections):
                i, j = divmod(int(pair), total)
                iou = intersection / (sizes[i] + sizes[j] - intersection)
                if iou >= (0.42 if classes[i] == classes[j] else 0.60):
                    edges.append((iou, i, j))
    for _, i, j in sorted(edges, reverse=True):
        left, right = find(i), find(j)
        if left != right and members[left].isdisjoint(members[right]):
            parent[right] = left
            members[left].update(members[right])
    clusters = {}
    for i in range(total):
        clusters.setdefault(find(i), []).append(i)

    floor = (bins == -2).sum(axis=1) + 0.05 * (bins[:, 0] == -2)
    wall = (bins == -1).sum(axis=1) + 0.05 * (bins[:, 0] == -1)
    semantic = (floor > wall).astype(np.uint8)
    instance = np.zeros(len(bins), np.int32)
    best = np.zeros(len(bins), np.float32)
    for instance_id, group in enumerate(clusters.values(), 1):
        if len(group) < 2:
            continue
        weights = [1.15 if views[i] == 0 else 1.0 for i in group]
        category = np.bincount([classes[i] for i in group], weights=weights).argmax()
        amount = sum((handles[views[i]] == i) * w for i, w in zip(group, weights))
        take = (amount >= len(group) * 0.5) & (amount > best)
        best[take] = amount[take]
        semantic[take], instance[take] = category, instance_id
    return semantic[restore], instance[restore]

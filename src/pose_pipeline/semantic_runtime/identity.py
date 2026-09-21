"""T1 global co-visibility association, ported from the sealed identity pilot.

The candidate/edge/ownership rules below retain the original thresholds. This
module has no experiment paths, GT access, model imports or installation work.
"""
from collections import defaultdict
from itertools import product
import numpy as np

CFG = {'min_mask_points': 30, 'min_common_positive_points': 30, 'full_iou_min': 0.25, 'covis_iou_min': 0.55, 'covis_cannot_link_max': 0.1, 'min_mutually_visible_positive_points': 30, 'min_distinct_support_frames': 2, 'min_output_points': 50, 'ownership_share': 0.65, 'ownership_margin': 0.15, 'max_known_stuff_fraction': 0.5, 'max_existing_instance_fraction_for_birth': 0.2, 'completion_known_owner_fraction_min': 0.8, 'completion_owner_observed_frames_min': 2, 'completion_owner_points_per_frame_min': 30, 'completion_new_points_max_fraction_of_existing_owner': 1.0}


def anchored_proposals(base, queries, seeds):
    """Select three-view mask alternatives using the first tracked object anchor.

    Queries contain eroded, depth-visible point sets for every SAM3 alternative.
    The first-view mask is the original point-prompt selection; later masks are
    selected jointly from alternatives passing the anchored support rule.
    """
    from .enhance import point_ids, validate_labels
    n = validate_labels(base)
    grouped = defaultdict(list)
    for row in queries:
        grouped[row['region']].append(row)
    observations = []
    for region, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row['view_index'])
        if [r['view_index'] for r in rows] != [0, 1, 2] or len({r['frame_id'] for r in rows}) != 3:
            raise ValueError('T1 requires exactly three distinct original views per region')
        seed = point_ids(seeds[str(region)], n)
        first = None
        for row in rows:
            visible = point_ids(row['visible'], n)
            anchor = overlap(first['points'], visible) if first is not None else np.empty(0, np.int64)
            effective_seed = anchor if len(anchor) >= 30 else seed
            vs = overlap(visible, effective_seed)
            choices = []
            for mi, candidate in enumerate(row['candidates']):
                points = point_ids(candidate['points'], n)
                if not np.isin(points, visible).all() or not np.isfinite(candidate['score']):
                    raise ValueError('invalid mask support or quality')
                count = len(overlap(points, vs))
                if count >= 10 and count / max(1, len(vs)) >= .5:
                    choices.append({'mask_index': mi, 'quality': float(candidate['score']),
                                    'points': points, 'visible': visible, 'frame': row['frame_id'],
                                    'coverage': count / max(1, len(vs)), 'source': 'original'})
            empty = {'mask_index': None, 'quality': 0., 'points': np.empty(0, np.int64),
                     'visible': visible, 'frame': row['frame_id'], 'coverage': 0., 'source': 'original'}
            if first is None:
                selected = next((c for c in choices if c['mask_index'] == row['chosen_mask_index']), empty)
                first = selected
                choices = [selected]
            else:
                choices = choices or [empty]
                selected = sorted(choices, key=lambda c: (-c['quality'], -c['coverage']))[0]
            observations.append({'record': row, 'choices': choices, 'original': selected})
    if not observations:
        return {k: v.copy() for k, v in base.items()}, {'accepted': [], 'no_queries': True}
    nodes, selection = choose(observations, True)
    labels, audit = graph_fusion(nodes, base, [], 'covis', False)
    audit['selection'] = selection
    return labels, audit

def overlap(a, b):
    return np.intersect1d(a, b, assume_unique=True)

def score_pair(a, b):
    shared = len(overlap(a['points'], b['points']))
    left = len(overlap(a['points'], b['visible']))
    right = len(overlap(b['points'], a['visible']))
    full = shared / max(1, len(a['points']) + len(b['points']) - shared)
    covis = shared / max(1, left + right - shared)
    evaluable = min(left, right) >= CFG['min_mutually_visible_positive_points']
    return {'shared': shared, 'left_visible': left, 'right_visible': right,
            'full_iou': full, 'covis_iou': covis, 'evaluable': evaluable,
            'cannot_link': bool(evaluable and covis <= CFG['covis_cannot_link_max'])}

def choose(observations, joint):
    by_region = defaultdict(list)
    for obs in observations: by_region[obs['record']['region']].append(obs)
    nodes, selection = [], []
    for region, rows in sorted(by_region.items()):
        rows.sort(key=lambda x: x['record']['view_index'])
        picked = [x['original'] for x in rows]
        if joint:
            best, best_key = None, None
            for combination in product([picked[0]], *[x['choices'] for x in rows[1:]]):
                scores = [score_pair(combination[i], combination[j]) for i,j in [(0,1),(0,2),(1,2)]]
                strong = sum(x['evaluable'] and x['covis_iou'] >= CFG['covis_iou_min'] and x['shared'] >= 30 for x in scores)
                agreement = sum(x['covis_iou'] for x in scores if x['evaluable'])
                key = (strong, agreement, sum(x['quality'] for x in combination), -sum(len(x['points']) for x in combination))
                if best_key is None or key > best_key: best, best_key = combination, key
            picked = list(best)
        for obs, candidate in zip(rows, picked):
            node = {**candidate, 'region': region, 'view': obs['record']['view_index'],
                    'query_name': obs['record']['name'], 'node_id': len(nodes)}
            nodes.append(node)
            selection.append({'region': region, 'view': node['view'], 'frame': node['frame'],
                              'original_mask': obs['original']['mask_index'], 'selected_mask': node['mask_index'],
                              'changed': node['mask_index'] != obs['original']['mask_index'],
                              'selected_points': len(node['points'])})
    return nodes, selection

def graph_fusion(nodes, base, original_objects, metric, complete):
    n = len(nodes); parents = list(range(n)); members = [{i} for i in range(n)]
    cannot = set(); edges = []; pair_rows = []
    for i in range(n):
        if len(nodes[i]['points']) < 30: continue
        for j in range(i+1, n):
            if len(nodes[j]['points']) < 30: continue
            p = score_pair(nodes[i], nodes[j]); pair_rows.append({'a':i, 'b':j, **p})
            if p['cannot_link']: cannot.add((i,j))
            score = p['full_iou'] if metric == 'full' else p['covis_iou']
            cutoff = CFG['full_iou_min'] if metric == 'full' else CFG['covis_iou_min']
            if score >= cutoff and p['shared'] >= 30 and (metric == 'full' or p['evaluable']):
                edges.append((score, p['shared'], i, j))
    def find(i):
        while parents[i] != i: parents[i] = parents[parents[i]]; i = parents[i]
        return i
    accepted_edges, vetoes = [], []
    for score, shared, i, j in sorted(edges, key=lambda x:(-x[0],-x[1],x[2],x[3])):
        a, b = find(i), find(j)
        if a == b: continue
        bad = next(((x,y) for x in members[a] for y in members[b] if (min(x,y),max(x,y)) in cannot), None)
        if bad: vetoes.append({'edge':[i,j], 'contradiction':list(bad)}); continue
        if a > b: a, b = b, a
        parents[b] = a; members[a] |= members[b]; members[b] = set()
        accepted_edges.append({'a':i, 'b':j, 'score':score, 'shared':shared})
    group_ids = [sorted(x) for x in members if x]
    base_names = {int(x['instance_id']):int(x['semantic_id']) for x in original_objects}
    total = np.zeros(len(base['instance']), np.uint16); proposals = []; rejected = []
    for group, ids in enumerate(group_ids):
        frames = defaultdict(list)
        for i in ids:
            if len(nodes[i]['points']): frames[nodes[i]['frame']].append(nodes[i]['points'])
        unions = {f: np.unique(np.concatenate(points)) for f, points in frames.items()}
        support = np.zeros(len(total), np.uint16)
        for points in unions.values(): support[points] += 1
        measured = np.flatnonzero(support >= 2)
        stuff = np.isin(base['semantic'][measured], [10,19]).mean() if len(measured) else 0.
        existing = (base['instance'][measured] > 0).mean() if len(measured) else 0.
        eligible = measured[(base['instance'][measured] == 0) & (base['semantic'][measured] == 0)]
        item = {'group': group, 'nodes': ids, 'regions': sorted(set(nodes[i]['region'] for i in ids)),
                'frames': sorted(unions), 'measured_points': len(measured),
                'eligible_points': len(eligible), 'stuff_fraction': float(stuff),
                'existing_fraction': float(existing), 'kind': 'birth', 'owner': None}
        reason = None
        if len(frames) < 2: reason = 'one_frame_only'
        elif stuff > .5: reason = 'mostly_known_stuff'
        elif len(eligible) < 50: reason = 'insufficient_unknown_points'
        elif existing > .2:
            reason = 'already_represented'
            owned = base['instance'][measured]; owned = owned[owned > 0]
            if complete and len(owned):
                owners, counts = np.unique(owned, return_counts=True)
                owner = int(owners[counts.argmax()]); share = counts.max()/len(owned)
                frame_support = sum(np.sum(base['instance'][points] == owner) >= 30 for points in unions.values())
                old_size = int(np.sum(base['instance'] == owner))
                item.update(owner=owner, owner_share=float(share), owner_frame_support=frame_support, owner_old_size=old_size)
                if share >= .8 and frame_support >= 2 and len(eligible) <= old_size and base_names.get(owner,0) not in [0,10,19]:
                    item['kind'] = 'complete'; reason = None
        if reason: rejected.append({**item, 'reason':reason}); continue
        votes = support[eligible]
        total[eligible] += votes
        proposals.append((item, eligible, votes))
    labels = {k: x.copy() for k,x in base.items()}; instance = labels['instance']; next_id = int(instance.max())+1
    top = np.zeros(len(total), np.uint16); second = top.copy(); winner = np.full(len(total), -1, np.int32)
    for j, (_, ids, votes) in enumerate(proposals):
        better = votes > top[ids]
        second[ids[better]] = top[ids[better]]; top[ids[better]] = votes[better]; winner[ids[better]] = j
        other = ~better; second[ids[other]] = np.maximum(second[ids[other]], votes[other])
    accepted = []; completed_per_owner = defaultdict(int)
    for j, (item, ids, votes) in enumerate(proposals):
        good = (winner[ids] == j) & (votes/np.maximum(1,total[ids]) >= .65) & ((votes.astype(float)-second[ids])/np.maximum(1,total[ids]) >= .15)
        ids = ids[good]
        if len(ids) < 50: rejected.append({**item,'reason':'ambiguous_ownership_or_small','owned_points':len(ids)}); continue
        k = item['owner'] if item['kind'] == 'complete' else next_id
        if item['kind'] == 'complete':
            limit = int(item['owner_old_size'] * CFG['completion_new_points_max_fraction_of_existing_owner'])
            if completed_per_owner[k] + len(ids) > limit:
                rejected.append({**item, 'reason': 'aggregate_owner_completion_cap', 'owned_points': len(ids)})
                continue
            completed_per_owner[k] += len(ids)
        if item['kind'] == 'birth': next_id += 1
        assert np.all(instance[ids] == 0)
        instance[ids] = k; accepted.append({**item, 'instance_id': k, 'new_points': len(ids)})
    return labels, {'accepted':accepted,'rejected':rejected,'edges':accepted_edges,'merge_vetoes':vetoes,
                    'pairs':pair_rows,'components':len(group_ids)}

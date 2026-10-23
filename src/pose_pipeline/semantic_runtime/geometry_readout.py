"""Final instance readout only, bound to DESIGN_AMENDMENT_V2.

Semantic/confidence remain byte-identical. No new object ID or inferred class.
Geometry groups link to existing final IDs only through complete owner lineage.
"""
import hashlib
import numpy as np

VERSION = 'semantic_abstention_geometry_final_readout_v1'


def _labels(value, n):
    if set(value) != {'instance', 'semantic', 'confidence'}:
        raise ValueError('unexpected label schema')
    for key, array in value.items():
        if np.asarray(array).shape != (n,):
            raise ValueError('label point domain differs')
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError('invalid label values')
    if any(np.asarray(value[k]).dtype.kind not in 'iu' for k in ('instance', 'semantic')):
        raise ValueError('labels must use integer identities')
    if np.any(value['confidence'] > 1):
        raise ValueError('confidence outside original domain')


def _hash(points, final_ids):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(points, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(final_ids, dtype=np.int64).tobytes())
    return h.hexdigest()


def apply_geometry_readout(old_off, final, safe_current, geometry_group_ids):
    n = len(old_off['instance'])
    _labels(old_off, n); _labels(final, n)
    safe = np.asarray(safe_current)
    geometry = np.asarray(geometry_group_ids)
    for array in (safe, geometry):
        if array.shape != (n,) or array.dtype.kind not in 'iu' or np.any(array < 0):
            raise ValueError('invalid geometry/current identity domain')
    if np.any((safe > 0) & (geometry == 0)):
        raise ValueError('emitted safe owner has no geometry group')
    # Full original Off positive fields are immutable through original stages.
    original_positive = old_off['instance'] > 0
    if not (np.array_equal(old_off['semantic'][original_positive], final['semantic'][original_positive])
            and np.array_equal(old_off['confidence'][original_positive], final['confidence'][original_positive])):
        raise ValueError('original final arm changed positive owner fields')
    candidate = ((old_off['instance'] == 0) & (final['instance'] == 0)
                 & (old_off['semantic'] == 0) & (final['semantic'] == 0) & (geometry > 0))
    result = {k: np.asarray(v).copy() for k, v in final.items()}
    links, recovered_parts = [], []
    for gid in np.unique(geometry[geometry > 0]):
        gid = int(gid)
        safe_points = np.flatnonzero((geometry == gid) & (safe > 0))
        source_ids = np.unique(safe[safe_points])
        row = dict(geometry_gid=gid, safe_anchor_points=len(safe_points),
                   candidate_points=int(np.count_nonzero(candidate & (geometry == gid))))
        reason = None
        if len(source_ids) == 0:
            reason = 'no_original_safe_output_owner'
        elif len(source_ids) != 1:
            reason = 'multiple_original_safe_output_owners'
        else:
            current_id = int(source_ids[0])
            # Check ALL points of this current ID, not one representative.
            current_points = np.flatnonzero(safe == current_id)
            if np.any(geometry[current_points] != gid):
                raise ValueError('safe current ID spans inconsistent geometry groups')
            off_ids = np.unique(old_off['instance'][current_points])
            row.update(original_current_id=current_id, original_current_points=len(current_points))
            if len(off_ids) != 1 or off_ids[0] <= 0:
                reason = 'current_to_complete_off_not_unique_positive'
            elif not np.all(old_off['instance'][current_points] == safe[current_points]):
                reason = 'guided_changed_original_safe_positive_owner'
            else:
                off_owner = int(off_ids[0])
                off_points = np.flatnonzero(old_off['instance'] == off_owner)
                final_ids = np.unique(final['instance'][off_points])
                row.update(old_off_owner_id=off_owner, old_off_owner_points=len(off_points),
                           whole_owner_lineage_sha256=_hash(off_points, final['instance'][off_points]),
                           observed_final_owner_ids=list(map(int, final_ids)))
                if len(final_ids) != 1 or final_ids[0] <= 0:
                    reason = 'whole_off_owner_final_lineage_not_unique_positive'
                else:
                    final_owner = int(final_ids[0])
                    row['final_owner_id'] = final_owner
                    selected = np.flatnonzero(candidate & (geometry == gid))
                    result['instance'][selected] = final_owner
                    row['recovered_points'] = len(selected)
                    if len(selected):
                        recovered_parts.append((selected, row.copy()))
        row['status'] = 'abstain' if reason else 'linked'
        row['reason'] = reason
        links.append(row)
    added = np.flatnonzero((final['instance'] == 0) & (result['instance'] > 0))
    unchanged = final['instance'] > 0
    if not np.array_equal(result['instance'][unchanged], final['instance'][unchanged]):
        raise RuntimeError('readout changed an existing final owner')
    if not all(np.array_equal(result[k], final[k]) for k in ('semantic', 'confidence')):
        raise RuntimeError('readout changed semantic/confidence')
    if len(added) and not np.isin(result['instance'][added], final['instance'][final['instance'] > 0]).all():
        raise RuntimeError('readout created a new object ID')
    fields = {k: [] for k in ('point_id', 'geometry_gid', 'original_current_id',
                              'old_off_owner_id', 'final_owner_id', 'whole_owner_lineage_sha256')}
    for points, row in recovered_parts:
        fields['point_id'].append(points)
        for key in fields:
            if key != 'point_id':
                fields[key].append(np.full(len(points), row[key], dtype='U64' if key.endswith('sha256') else np.int64))
    provenance = {k: np.concatenate(v) if v else np.empty(0, dtype='U64' if k.endswith('sha256') else np.int64)
                  for k, v in fields.items()}
    if len(added):
        order = np.argsort(provenance['point_id'])
        provenance = {k: v[order] for k, v in provenance.items()}
        if not np.array_equal(provenance['point_id'], added):
            raise RuntimeError('point provenance does not cover additions exactly once')
    audit = dict(version=VERSION, N=n, GT_read=False, new_instance_ids=False,
                 original_positive_owners_preserved=True, semantic_unchanged=True,
                 confidence_unchanged=True, original_pipeline_changed=False,
                 candidate_points=int(candidate.sum()), added_points=len(added),
                 geometry_groups=len(links), linked_geometry_groups=sum(r['status'] == 'linked' for r in links),
                 no_anchor_geometry_groups=sum(r['reason'] == 'no_original_safe_output_owner' for r in links),
                 no_anchor_candidate_points=sum(r['candidate_points'] for r in links if r['reason'] == 'no_original_safe_output_owner'),
                 lineage_abstain_groups=sum(r['status'] == 'abstain' and r['reason'] != 'no_original_safe_output_owner' for r in links),
                 lineage_abstain_candidate_points=sum(r['candidate_points'] for r in links if r['status'] == 'abstain' and r['reason'] != 'no_original_safe_output_owner'),
                 links=links)
    return result, audit, provenance

"""Hash-bound terminal owner readout; shared by writers and artifact consumers."""
from pathlib import Path
import hashlib
import json
import numpy as np
from ..contracts import sha256_file
from .geometry_readout import VERSION, apply_geometry_readout, _labels

POLICY = 'owner-lineage-v1'
SCHEMA = 'revemap.geometry_readout.v1'
FILES = {'geometry_base': 'map_labels_pre_geometry_readout.npz',
         'geometry_capture': 'GEOMETRY_CAPTURE.npz',
         'geometry_lineage': 'GEOMETRY_POINT_LINEAGE.npz',
         'geometry_audit': 'GEOMETRY_READOUT.json',
         'geometry_contract': 'GEOMETRY_CONTRACT.json'}


def xyz_hash(xyz):
    return hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _same(expected, actual, what):
    if set(expected) != set(actual) or any(
            expected[k].dtype != actual[k].dtype
            or expected[k].shape != actual[k].shape
            or not np.array_equal(expected[k], actual[k]) for k in expected):
        raise ValueError(what + ' differs from exact terminal readout')


def artifact_files(root):
    return {k: Path(root) / v for k, v in FILES.items()}


def save_contract(root, xyz, base, final, capture, audit, lineage, provenance, *, source_files=None):
    """Create-only sidecars after final labels exist; never replace an old run."""
    root = Path(root)
    files = artifact_files(root)
    if any(p.exists() for p in files.values()):
        raise ValueError('geometry readout sidecar already exists')
    np.savez_compressed(files['geometry_base'], **base)
    np.savez_compressed(files['geometry_capture'], **capture)
    np.savez_compressed(files['geometry_lineage'], **lineage)
    files['geometry_audit'].write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    source_paths = {k: provenance[k] for k in ('manifest', 'trajectory')}
    if set(source_files or {}) - {'mapping_result', 'sam3_frames', 'raw_inputs', 'baseline_cloud'}:
        raise ValueError('unexpected geometry source file role')
    source_paths.update(source_files or {})
    sources = {k: {'path': str(Path(p).resolve()), 'sha256': sha256_file(Path(p))}
               for k, p in source_paths.items()}
    contract = {'schema': SCHEMA, 'policy': POLICY, 'version': VERSION,
                'N': len(xyz), 'geometry_xyz_sha256': xyz_hash(xyz),
                'readout_source_sha256': sha256_file(Path(__file__).with_name('geometry_readout.py')),
                'capture_source_sha256': sha256_file(Path(__file__).with_name('multiview.py')),
                'source_provenance': sources,
                'scene_id': provenance['scene_id'], 'dataset': provenance['dataset'],
                'files': {k: {'path': p.name, 'sha256': sha256_file(p)}
                          for k, p in files.items() if k != 'geometry_contract'},
                'final_labels': {'path': 'map_labels.npz', 'sha256': sha256_file(root / 'map_labels.npz')},
                'label_arrays': {k: {'dtype': v.dtype.str, 'shape': list(v.shape),
                                    'sha256': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
                                 for k, v in final.items()}}
    files['geometry_contract'].write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
    return files['geometry_contract']


def validate_contract(contract_path, xyz, final, *, provenance=None, artifact_paths=None):
    """Recompute the frozen readout and require all sidecars, labels and binding.

    Hashes provide integrity within a sealed run, not signatures for untrusted
    fully rewritten bundles. No GT, model, geometry association or network calls.
    """
    path = Path(contract_path).resolve(); root = path.parent
    contract = json.loads(path.read_text())
    if (contract.get('schema') != SCHEMA or contract.get('policy') != POLICY
            or contract.get('version') != VERSION or contract.get('N') != len(xyz)
            or contract.get('geometry_xyz_sha256') != xyz_hash(xyz)):
        raise ValueError('geometry readout contract/point order mismatch')
    for key, filename in [('readout_source_sha256', 'geometry_readout.py'),
                          ('capture_source_sha256', 'multiview.py')]:
        if contract.get(key) != sha256_file(Path(__file__).with_name(filename)):
            raise ValueError('geometry readout source identity mismatch')
    paths = artifact_files(root)
    expected = set(FILES) - {'geometry_contract'}
    if set(contract.get('files', {})) != expected:
        raise ValueError('geometry readout sidecar inventory differs')
    for key in expected:
        entry = contract['files'][key]; p = paths[key]
        if entry.get('path') != p.name or not p.is_file() or sha256_file(p) != entry.get('sha256'):
            raise ValueError('geometry sidecar digest mismatch: ' + key)
    final_path = root / 'map_labels.npz'
    entry = contract.get('final_labels', {})
    if entry.get('path') != final_path.name or sha256_file(final_path) != entry.get('sha256'):
        raise ValueError('geometry final labels digest mismatch')
    if artifact_paths is not None:
        if any(key not in artifact_paths or Path(artifact_paths[key]).resolve() != paths[key].resolve()
               for key in FILES):
            raise ValueError('artifact inventory does not bind every geometry sidecar')
    sources = contract.get('source_provenance', {})
    if not {'manifest', 'trajectory'} <= set(sources) or set(sources) - {
            'manifest', 'trajectory', 'mapping_result', 'sam3_frames', 'raw_inputs', 'baseline_cloud'}:
        raise ValueError('geometry source provenance missing')
    for key, entry in sources.items():
        p = Path(entry['path'])
        if not p.is_file() or sha256_file(p) != entry.get('sha256'):
            raise ValueError('geometry source provenance digest mismatch: ' + key)
        if provenance is not None and key in ('manifest', 'trajectory') and Path(provenance[key]).resolve() != p.resolve():
            raise ValueError('geometry source provenance path differs: ' + key)
    if provenance is not None and any(contract.get(k) != provenance[k] for k in ('scene_id', 'dataset')):
        raise ValueError('geometry source scene identity differs')
    disk_final = load_arrays(final_path); _same(disk_final, final, 'consumer labels')
    _labels(final, len(xyz))
    for k, v in final.items():
        metadata = {'dtype': v.dtype.str, 'shape': list(v.shape),
                    'sha256': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
        if contract.get('label_arrays', {}).get(k) != metadata:
            raise ValueError('geometry label array binding differs: ' + k)
    base = load_arrays(paths['geometry_base']); _labels(base, len(xyz))
    if np.any((base['instance'] > 0) & (base['semantic'] == 0)):
        raise ValueError('pre-readout positive owner cannot have unknown semantic')
    capture = load_arrays(paths['geometry_capture'])
    if set(capture) != {'object_id', 'safe_current', 'eligible_group_ids'}:
        raise ValueError('unexpected geometry capture schema')
    geometry, safe, groups = (capture[k] for k in ('object_id', 'safe_current', 'eligible_group_ids'))
    if (groups.ndim != 1 or groups.dtype.kind not in 'iu'
            or not np.array_equal(groups, np.arange(1, len(groups) + 1))
            or not np.isin(geometry[geometry > 0], groups).all()):
        raise ValueError('geometry group outside original eligible group universe')
    derived, audit, lineage = apply_geometry_readout(base, base, safe, geometry)
    _same(derived, final, 'final labels')
    _same(lineage, load_arrays(paths['geometry_lineage']), 'point lineage')
    if json.loads(paths['geometry_audit'].read_text()) != audit:
        raise ValueError('geometry audit differs from frozen readout')
    return {'base': base, 'audit': audit, 'contract': contract}


def from_artifacts(paths, xyz, final):
    """Detect readout through the required result declaration, never a boolean."""
    report = json.loads(Path(paths['result']).read_text())
    policy = report.get('geometry_readout', 'off')
    has_sidecar = any(k in paths for k in FILES)
    if policy == 'off':
        if has_sidecar:
            raise ValueError('off artifact cannot declare geometry sidecars')
        if np.any((final['instance'] > 0) & (final['semantic'] == 0)):
            raise ValueError('unknown positive owner requires sealed geometry contract')
        return None
    if policy != POLICY or 'geometry_contract' not in paths:
        raise ValueError('unsupported or unbound geometry readout artifact')
    provenance = report.get('provenance')
    if provenance is None:
        if not {'input_manifest', 'trajectory'} <= paths.keys():
            raise ValueError('GUI geometry readout has no bound source provenance')
        from ..contracts import load_manifest
        m = load_manifest(Path(paths['input_manifest']), require_files=False)
        provenance = {'manifest': paths['input_manifest'], 'trajectory': paths['trajectory'],
                      'scene_id': m.sequence_id, 'dataset': m.dataset}
    return validate_contract(paths['geometry_contract'], xyz, final,
                             provenance=provenance, artifact_paths=paths)

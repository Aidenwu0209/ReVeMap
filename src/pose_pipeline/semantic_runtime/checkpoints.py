"""Verified reuse of completed mapping and SAM3 stages in a new run directory.

A failed attempt is never resumed in place. Missing/stale checkpoints cause a
fresh stage execution; partially written stages never qualify for reuse.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .common import SOURCE_ROOT, read, sha, write

STAGES = {'mapping': 'mapping', 'sam3': 'semantic'}


def fingerprint_runtime(config):
    """Include external model code/weights and interpreter package versions."""
    files = set()
    for key in ('provider_root', 'sam3_source'):
        root = Path(config[key]).resolve()
        files.update(p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts
                     and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.yaml', '.yml', '.toml', '.so'))
    files.add(Path(config['provider_root']) / 'pretrained/droid.pth')
    files.add(Path(config['sam3_checkpoint']))
    interpreters = {}
    probe = ('import importlib.metadata as m,json,sys; '
             'print(json.dumps({"python":sys.version,"packages":sorted('
             '(d.metadata.get("Name",""),d.version) for d in m.distributions())}))')
    for key in ('cpu_python', 'gpu_python', 'sam3_python'):
        python = str(Path(config[key]).absolute())
        if python not in interpreters:
            interpreters[python] = json.loads(subprocess.check_output(
                [python, '-c', probe], text=True, timeout=30))
            files.add(Path(python).resolve())
    return {'files': {str(p.resolve()): sha(p) for p in sorted(files)},
            'interpreters': interpreters}


def context(inputs, config, options, *, runtime_fingerprint=None):
    return {'inputs': inputs, 'runtime': config, 'options': options,
            'source': {str(p.relative_to(SOURCE_ROOT)): sha(p)
                       for p in sorted(SOURCE_ROOT.rglob('*')) if p.is_file()
                       and '__pycache__' not in p.parts and p.suffix in ('.py', '.json')},
            'external_runtime': runtime_fingerprint if runtime_fingerprint is not None
                                else fingerprint_runtime(config)}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _stage_files(directory):
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('stage checkpoint cannot contain symbolic links')
        if path.is_file():
            files[str(path.relative_to(directory))] = sha(path)
    return files


def _completed(stage, directory):
    if stage == 'mapping':
        receipt = read(directory / 'mapping_result.json')
        if receipt.get('status') != 'completed':
            raise ValueError('mapping has no successful receipt')
        for key, digest in (('manifest', 'manifest_sha256'), ('trajectory', 'trajectory_sha256'),
                            ('final_cloud', 'final_cloud_sha256')):
            if sha(receipt[key]) != receipt[digest]:
                raise ValueError('mapping checkpoint source does not match: ' + key)
        if receipt.get('mesh'):
            mesh = Path(receipt['mesh']).resolve()
            if not mesh.is_relative_to(directory.resolve()) or sha(mesh) != receipt.get('mesh_sha256'):
                raise ValueError('mapping checkpoint surface does not match')
    else:
        receipt = read(directory / 'COMPLETE.json')
        frames = read(directory / 'FRAMES.json')
        if (receipt.get('status') != 'completed' or len(frames) != receipt.get('selected_frames')
                or not frames or len({r['frame_id'] for r in frames}) != len(frames)):
            raise ValueError('SAM3 has no complete frame inventory')
        if receipt.get('view_plan_sha256') is not None:
            if sha(directory / 'VIEW_PLAN.json') != receipt['view_plan_sha256']:
                raise ValueError('SAM3 checkpoint view plan changed')
        for row in frames:
            if sha(directory / 'frames' / f'{row["frame_id"]:06}.npz') != row['mask_sha256']:
                raise ValueError('SAM3 checkpoint mask changed')


class CheckpointStore:
    def __init__(self, output, expected_context, resume_from=None):
        self.output = Path(output).resolve()
        self.expected_context = expected_context
        self.signature = _digest(expected_context)
        self.source = Path(resume_from).resolve() if resume_from else None
        if self.source and (self.source == self.output or self.source.is_relative_to(self.output)
                            or self.output.is_relative_to(self.source)):
            raise ValueError('resume source and new output must be separate run directories')
        self.audit = []
        write(self.output / 'CHECKPOINT_CONTEXT.json', expected_context)

    def verify_context(self):
        # A run may fail after this stage, so the final run-wide check is too
        # late to decide whether its completed stage is safe to reuse.
        inputs = self.expected_context.get('inputs', {})
        if isinstance(inputs, dict):
            for path, expected in inputs.items():
                if sha(path) != expected:
                    raise ValueError('checkpoint input changed before stage seal: ' + path)
        if 'external_runtime' in self.expected_context:
            for name, key in [('runtime.json', 'runtime'), ('CONFIG.json', 'options'), ('INPUTS.json', 'inputs')]:
                if read(self.output / name) != self.expected_context[key]:
                    raise ValueError('checkpoint execution record changed before stage seal: ' + name)
            current = context(inputs, self.expected_context['runtime'], self.expected_context['options'])
            if current != self.expected_context:
                raise ValueError('checkpoint code or runtime changed before stage seal')

    def seal(self, stage, *, reused_from=None):
        self.verify_context()
        directory = self.output / STAGES[stage]
        _completed(stage, directory)
        files = _stage_files(directory)
        payload = {'schema': 'revemap.stage_checkpoint.v1', 'status': 'completed', 'stage': stage,
                   'context_sha256': self.signature, 'files': files, 'reused_from': reused_from}
        write(self.output / 'checkpoints' / (stage + '.json'),
              {**payload, 'payload_sha256': _digest(payload)})

    def restore(self, stage, *, view_plan=None):
        if self.source is None:
            return False
        checkpoint = self.source / 'checkpoints' / (stage + '.json')
        source_dir = self.source / STAGES[stage]
        destination = self.output / STAGES[stage]
        # Copying to a temporary sibling keeps a failed copy out of stage outputs.
        temporary = self.output / ('.reuse-' + stage)
        try:
            payload = read(checkpoint)
            digest = payload.pop('payload_sha256')
            if (digest != _digest(payload) or payload.get('schema') != 'revemap.stage_checkpoint.v1'
                    or payload.get('status') != 'completed' or payload.get('stage') != stage):
                raise ValueError('invalid checkpoint seal')
            if payload.get('context_sha256') != self.signature:
                raise ValueError('input, options, code or runtime changed')
            if _digest(read(self.source / 'CHECKPOINT_CONTEXT.json')) != self.signature:
                raise ValueError('checkpoint context changed')
            for relative, expected in payload['files'].items():
                path = source_dir / relative
                if (Path(relative).is_absolute() or '..' in Path(relative).parts
                        or not path.resolve().is_relative_to(source_dir.resolve())
                        or path.is_symlink() or sha(path) != expected):
                    raise ValueError('checkpoint output changed: ' + relative)
            _completed(stage, source_dir)
            temporary.mkdir(exist_ok=False)
            # Only sealed files are copied: later VLM outputs are never included
            # in a SAM3 checkpoint, even when the previous run subsequently failed.
            for relative in payload['files']:
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_dir / relative, target)
            if _stage_files(temporary) != payload['files']:
                raise ValueError('checkpoint changed while copying')
            if stage == 'mapping':
                receipt = read(temporary / 'mapping_result.json')
                for key in ('manifest', 'trajectory', 'final_cloud', 'mesh'):
                    if key not in receipt:
                        continue
                    path = Path(receipt[key]).resolve()
                    if path.is_relative_to(source_dir):
                        receipt[key] = str(destination / path.relative_to(source_dir))
                write(temporary / 'mapping_result.json', receipt)
                nested = temporary / 'fusion/refusion_result.json'
                if nested.exists():
                    refusion = read(nested)
                    for key in ('cloud', 'manifest', 'trajectory', 'mesh'):
                        if key in refusion and Path(refusion[key]).resolve().is_relative_to(source_dir):
                            refusion[key] = str(destination / Path(refusion[key]).resolve().relative_to(source_dir))
                    write(nested, refusion)
            else:
                if view_plan is not None:
                    original = read(temporary / 'VIEW_PLAN.json')
                    current = read(view_plan)
                    if {k: v for k, v in original.items() if k != 'trajectory'} != {k: v for k, v in current.items() if k != 'trajectory'}:
                        raise ValueError('SAM3 view plan differs beyond its rebound trajectory path')
                    if sha(current['trajectory']) != current['trajectory_sha256']:
                        raise ValueError('SAM3 rebound trajectory changed')
                    write(temporary / 'VIEW_PLAN.json', current)
                    receipt = read(temporary / 'COMPLETE.json')
                    receipt['reused_original_plan_sha256'] = receipt['view_plan_sha256']
                    receipt['view_plan_sha256'] = sha(temporary / 'VIEW_PLAN.json')
                    write(temporary / 'COMPLETE.json', receipt)
                tasks = read(temporary / 'CROP_TASKS.json')
                for task in tasks:
                    for crop in task['crops']:
                        path = Path(crop['file']).resolve()
                        if not path.is_relative_to(source_dir):
                            raise ValueError('SAM3 crop is outside its completed stage')
                        crop['file'] = str(destination / path.relative_to(source_dir))
                write(temporary / 'CROP_TASKS.json', tasks)
            temporary.rename(destination)
            self.seal(stage, reused_from=str(checkpoint))
            self.audit.append({'stage': stage, 'reused': True, 'checkpoint': str(checkpoint)})
            return True
        except (OSError, ValueError, KeyError, TypeError) as error:
            if temporary.exists():
                shutil.rmtree(temporary)
            # Existing destination means a post-copy seal failed; do not silently
            # consume it or overwrite it with another stage execution.
            if destination.exists():
                raise
            self.audit.append({'stage': stage, 'reused': False, 'reason': str(error)})
            return False
        finally:
            write(self.output / 'RESUME.json', {'source': str(self.source), 'stages': self.audit,
                                              'scope': 'completed mapping/SAM3 stages only; later stages rerun'})

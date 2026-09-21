"""Regression contracts for optional naming, independent evidence and scheduling."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pose_pipeline.semantic_runtime.common import parse_label, registry, selected_frames, write
from pose_pipeline.semantic_runtime.enhance import grounded_names, name_consensus, verified_fill
from pose_pipeline.semantic_runtime.pipeline import Processes


def labels(n=200):
    return {'semantic': np.zeros(n, np.int32), 'instance': np.zeros(n, np.int32),
            'confidence': np.zeros(n, np.float32)}


def pair_inputs():
    old = labels()
    old['instance'][0:30] = 1
    old['semantic'][0:30] = 3
    base = {k: v.copy() for k, v in old.items()}
    base['instance'][40:80] = 2
    base['instance'][80:120] = 3
    points = np.arange(40, 150)
    views = [{'pair': [2, 3], 'frame_id': f, 'visible': np.arange(200),
              'left': [{'score': .9, 'points': points}], 'right': [{'score': .9, 'points': points}]} for f in [5, 25]]
    return old, base, views


def test_p2_requires_two_views_and_preserves_originals():
    old, base, views = pair_inputs()
    unchanged, audit = verified_fill(base, old, views[:1])
    assert np.array_equal(unchanged['instance'], base['instance'])
    result, audit = verified_fill(base, old, views)
    assert audit['accepted_pairs'] == [[2, 3]] and audit['added_points'] == 30
    assert np.all(result['instance'][40:150] == 2)
    for field in ['semantic', 'confidence']:
        assert np.array_equal(result[field], old[field])
    assert np.array_equal(result['instance'][:30], old['instance'][:30])


def test_duplicates_cannot_create_second_view():
    old, base, views = pair_inputs()
    with pytest.raises(ValueError, match='duplicate original frame'):
        verified_fill(base, old, [views[0], views[0]])


def test_one_prompt_leaking_does_not_verify_pair():
    old, base, views = pair_inputs()
    for row in views:
        row['right'] = [{'score': .99, 'points': np.arange(80, 120)}]
    result, audit = verified_fill(base, old, views)
    assert not audit['accepted_pairs']
    assert np.array_equal(result['instance'], base['instance'])


def naming_inputs():
    base = labels(160)
    base['instance'][:40] = 1
    base['semantic'][:40] = 3
    base['instance'][40:120] = 2
    names = [{'instance_id': 2, 'frame_id': f, 'role': 'fusion', 'crop_mode': 'context', 'raw_response': 'piano'} for f in [5, 25, 45]]
    points = np.arange(40, 120)
    grounds = [{'instance_id': 2, 'frame_id': f, 'role': 'fusion', 'label': 'piano',
                'visible': np.arange(160), 'reference': points,
                'candidates': [{'score': .9, 'points': points}]} for f in [5, 25]]
    return base, names, grounds


def test_multiview_name_needs_grounding_and_keeps_unknown():
    base, names, grounds = naming_inputs()
    classes = {'0': 'unknown', '3': 'chair'}
    partial, _, _, audit = grounded_names(base, classes, names, grounds[:1])
    assert not audit and np.array_equal(partial['semantic'], base['semantic'])
    final, classes, strength, audit = grounded_names(base, classes, names, grounds)
    assert len(audit) == 1 and classes[str(final['semantic'][40])] == 'piano'
    assert np.array_equal(final['instance'], base['instance'])
    assert np.array_equal(final['confidence'], base['confidence'])
    assert np.all(final['semantic'][:40] == 3) and np.all(final['semantic'][120:] == 0)
    assert np.isclose(strength[40], 2/3)


def test_heldout_and_duplicate_naming_votes_do_not_count():
    base, names, grounds = naming_inputs()
    for row in names[1:]:
        row['role'] = 'validation'
    assert name_consensus(names) == {}
    with pytest.raises(ValueError, match='duplicate context'):
        name_consensus([names[0], names[0]])


def test_bad_grounding_or_unknown_vocab_fails_closed():
    base, names, grounds = naming_inputs()
    grounds[0]['frame_id'] = 999
    with pytest.raises(ValueError, match='outside the naming view plan'):
        grounded_names(base, {'0': 'unknown', '3': 'chair'}, names, grounds)
    grounds[0]['frame_id'] = 5
    grounds[0]['candidates'][0]['points'] = np.arange(160)
    final, _, _, audit = grounded_names(base, {'0': 'unknown', '3': 'chair'}, names, grounds)
    assert not audit and np.array_equal(final['semantic'], base['semantic'])


def test_all_tested_families_have_interfaces():
    ids = registry()
    expected = ['qwen3vl_2b_bf16', 'qwen3vl_8b_nf4_44', 'qwen35_9b_nf4_44',
                'gemma4_e2b_qat_q4', 'minicpmv46_bf16_4x', 'minicpmv4_nf4',
                'internvl35_1b_bf16', 'smolvlm2_22b_bf16', 'mage_nf4', 'joy_nf4',
                'glm53flash_api', 'deepseek_v41_flash_api', 'paddleocr_vl16_api',
                'qwen3vl_2b_bf16_min65536', 'none']
    assert set(expected) <= set(ids)
    assert ids['paddleocr_vl16_api']['role'] == 'ocr_evidence_only'
    for mid, row in ids.items():
        if row['kind'] not in ('none', 'api', 'ocr'):
            assert (len(row['revision']) == 40 or row.get('provenance') == 'local_checkpoint_hashes_only')
            assert '/' in row['repo']


def test_stride_keeps_first_and_last_without_duplicates():
    frames = [SimpleNamespace(frame_id=i) for i in range(12)]
    assert [f.frame_id for f in selected_frames(frames, 5)] == [0, 5, 10, 11]
    assert len(selected_frames(frames, 1)) == 12
    with pytest.raises(ValueError):
        selected_frames(frames, 0)


def test_none_never_imports_or_loads_model(tmp_path):
    from pose_pipeline.semantic_runtime.worker import name_tasks
    from pose_pipeline.semantic_runtime.common import sha
    image = tmp_path / 'image.bin'
    image.write_bytes(b'No GPU or model needed')
    name_tasks([{'task_id': 0, 'crops': [{'file': str(image), 'sha256': sha(image)}]}],
               'none', {}, tmp_path / 'result')
    result = json.loads((tmp_path/'result/COMPLETE.json').read_text())
    assert result['crops_executed'] == 0


def test_api_sends_image_and_does_not_log_token(monkeypatch, tmp_path):
    from pose_pipeline.semantic_runtime.vlm import APINamer
    import requests
    monkeypatch.setenv('FAKE_TEST_KEY', 'test-only-secret')
    spec = {'id': 'test', 'role': 'object_naming', 'endpoint': 'https://example.com/v1/chat/completions',
            'token_env': 'FAKE_TEST_KEY', 'model': 'glm-5.3-flash', 'max_new_tokens': 256}
    seen = []
    def post(url, **kw):
        seen.append(kw)
        return SimpleNamespace(status_code=200, json=lambda: {'choices': [{'message': {'content': 'piano'}}]})
    monkeypatch.setattr(requests, 'post', post)
    image = tmp_path/'crop.png'
    image.write_bytes(b'image-bytes')
    model = APINamer(spec, {})
    answer = model.infer(image)
    assert answer['label'] == 'piano'
    assert seen[0]['json']['messages'][0]['content'][0]['image_url']['url'].startswith('data:image/png;base64,')
    assert 'test-only-secret' not in json.dumps(model.audit)
    assert seen[0]['json']['reasoning_effort'] == 'low'  # Retain GLM contract.


def test_deepseek_preserves_verified_non_thinking_image_payload(monkeypatch, tmp_path):
    from pose_pipeline.semantic_runtime.common import model_spec, PROMPT
    from pose_pipeline.semantic_runtime.vlm import create_namer
    import base64
    import requests
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-only-deepseek-secret')
    seen = []
    def post(url, **kw):
        seen.append((url, kw))
        return SimpleNamespace(status_code=200, json=lambda: {
            'model': 'deepseek-flash', 'system_fingerprint': 'test-fingerprint',
            'choices': [{'finish_reason': 'stop', 'message': {'content': 'shipping container'}}]})
    monkeypatch.setattr(requests, 'post', post)
    image = tmp_path / 'crop.png'
    image.write_bytes(b'original-image-bytes')
    result = create_namer('deepseek_v41_flash_api').infer(image)
    assert seen[0][0] == model_spec('deepseek_v41_flash_api')['endpoint']
    assert seen[0][1]['json'] == {
        'model': 'deepseek-flash', 'temperature': 0, 'max_tokens': 24,
        'thinking': {'type': 'disabled'}, 'stream': False,
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                base64.b64encode(image.read_bytes()).decode(), 'detail': 'original'}},
            {'type': 'text', 'text': PROMPT}]}]}
    assert seen[0][1]['allow_redirects'] is False
    assert result['label'] == 'shipping container'
    assert result['system_fingerprint'] == 'test-fingerprint'
    assert result['reasoning_content_present'] is False
    assert 'test-only-deepseek-secret' not in json.dumps(result)


@pytest.mark.parametrize('status', [401, 402, 500])
def test_deepseek_failure_does_not_return_a_name(monkeypatch, tmp_path, status):
    from pose_pipeline.semantic_runtime.vlm import create_namer
    import requests
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-only-deepseek-secret')
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: SimpleNamespace(status_code=status))
    path = tmp_path / 'crop.png'
    path.write_bytes(b'image')
    with pytest.raises(RuntimeError, match=f'API HTTP {status}'):
        create_namer('deepseek_v41_flash_api').infer(path)


@pytest.mark.parametrize('fault', [None, 'missing_pixels', 'wrong_grid'])
def test_mage_encodes_pixels_and_rejects_text_only_inputs(tmp_path, fault):
    """Exercise the actual adapter without loading GPU weights or installing Torch."""
    from contextlib import nullcontext
    from PIL import Image
    from pose_pipeline.semantic_runtime.vlm import TransformersNamer
    class Tensor(np.ndarray):
        def numel(self): return self.size
        def is_floating_point(self): return self.dtype.kind == 'f'
        def to(self, *a): return self
    def tensor(x): return np.asarray(x).view(Tensor)
    class Batch(dict):
        def to(self, device):
            assert device == 'cuda'
            return self
    class Processor:
        spatial_merge_size = 2
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda s: 99)
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['tokenize'] is False
            return 'image placeholder and naming prompt'
        def __call__(self, *, text, images, return_tensors):
            assert len(images) == 1 and images[0].size == (32, 24)
            assert images[0].getpixel((0, 0)) == (255, 0, 0)
            batch = Batch(input_ids=tensor([[1, 99, 2]]), pixel_values=tensor([[1.]]),
                          image_grid_thw=tensor([[1, 2, 2]]), patch_positions=tensor([[0, 0]]))
            if fault == 'missing_pixels': del batch['pixel_values']
            if fault == 'wrong_grid': batch['image_grid_thw'] = tensor([[1, 4, 4]])
            return batch
        def batch_decode(self, output, **kwargs): return ['chair']
    generated = []
    def generate(**kwargs):
        generated.append(kwargs)
        assert kwargs['pixel_values'].numel() > 0
        return tensor([[1, 99, 2, 3]])
    namer = object.__new__(TransformersNamer)
    namer.spec = {'kind': 'mage_custom', 'model': 'mage', 'max_new_tokens': 24}
    namer.processor, namer.model, namer.dtype = Processor(), SimpleNamespace(generate=generate), 'bf16'
    namer.torch = SimpleNamespace(Tensor=Tensor, inference_mode=nullcontext,
        cuda=SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 0))
    path = tmp_path / 'image.png'
    Image.new('RGB', (32, 24), 'red').save(path)
    if fault:
        with pytest.raises(RuntimeError, match='Mage image'):
            namer.infer(path)
        assert not generated
    else:
        result = namer.infer(path)
        assert result['label'] == 'chair' and result['image_inputs_verified']
        assert result['image_tokens'] == 1 and len(generated) == 1


def test_ocr_jobs_keep_text_separate_and_do_not_forward_auth(monkeypatch, tmp_path):
    from pose_pipeline.semantic_runtime.vlm import OCRNamer
    import requests
    monkeypatch.setenv('FAKE_OCR_KEY', 'test-only-ocr-secret')
    spec = {'id': 'ocr-test', 'role': 'ocr_evidence_only', 'endpoint': 'https://ocr.example/jobs',
            'token_env': 'FAKE_OCR_KEY', 'model': 'PaddleOCR-VL-1.6', 'poll_timeout': 2}
    submitted, fetched = [], []
    def post(url, **kwargs):
        submitted.append(kwargs)
        assert kwargs['files']['file'][1].read() == b'crop'
        return SimpleNamespace(status_code=200, json=lambda: {'code': 0, 'data': {'jobId': 'job-1'}})
    def get(url, **kwargs):
        fetched.append((url, kwargs))
        if url.endswith('/job-1'):
            return SimpleNamespace(status_code=200, json=lambda: {'code': 0, 'data': {
                'state': 'done', 'resultUrl': {'jsonUrl': 'https://results.example/result.jsonl'}}})
        page = {'result': {'layoutParsingResults': [{'prunedResult': {'parsing_res_list': [
            {'block_content': 'printed words'}]}}]}}
        return SimpleNamespace(text=json.dumps(page), raise_for_status=lambda: None)
    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(requests, 'get', get)
    path = tmp_path/'crop.png'
    path.write_bytes(b'crop')
    result = OCRNamer(spec, {}).infer(path)
    assert result['ocr_text'] == 'printed words'
    assert result['label'] == 'unknown' and result['valid'] is False
    assert submitted[0]['data']['model'] == 'PaddleOCR-VL-1.6'
    assert 'headers' not in fetched[-1][1]
    assert 'test-only-ocr-secret' not in json.dumps(result)


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process group supervisor')
def test_worker_failure_stops_instead_of_silently_succeeding(tmp_path):
    jobs = Processes(tmp_path, timeout=3)
    job = jobs.launch('failure', [sys.executable, '-c', 'raise SystemExit(4)'])
    try:
        with pytest.raises(RuntimeError, match='failure failed'):
            jobs.wait(job)
    finally:
        jobs.close()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process group supervisor')
def test_worker_timeout_terminates_owned_process(tmp_path):
    jobs = Processes(tmp_path, timeout=.1)
    job = jobs.launch('timeout', [sys.executable, '-c', 'import time; time.sleep(20)'])
    try:
        with pytest.raises(TimeoutError):
            jobs.wait(job)
    finally:
        jobs.close()
    assert job[1].poll() is not None

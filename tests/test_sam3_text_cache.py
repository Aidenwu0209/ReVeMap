from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from pose_pipeline.sam3_text_cache import cached_text_features


class Backbone:
    training = False

    def __init__(self):
        self.calls = 0

    def forward_text(self, captions, input_boxes=None, additional_text=None, device='cpu'):
        self.calls += 1
        return {'features': torch.tensor([len(captions[0])], device=device, dtype=torch.float32)}


def processor(backbone):
    return SimpleNamespace(model=SimpleNamespace(backbone=backbone))


def test_repeated_text_is_identical_and_downstream_mutation_is_isolated():
    backbone = Backbone()
    with cached_text_features(processor(backbone)) as stats, torch.inference_mode():
        first = backbone.forward_text(['chair'], device='cpu')
        first['features'].zero_()
        second = backbone.forward_text(['chair'], device='cpu')
        assert second['features'].item() == 5
        second['features'].zero_()
        assert backbone.forward_text(['chair'], device='cpu')['features'].item() == 5
        assert stats == {'hits': 2, 'misses': 1, 'entries': 1, 'bytes': 4}
    assert backbone.calls == 1
    assert 'forward_text' not in vars(backbone)


@pytest.mark.parametrize('limit', [{'max_entries': 1}, {'max_bytes': 4}])
def test_cache_is_bounded_and_uncached_prompts_still_compute(limit):
    backbone = Backbone()
    with cached_text_features(processor(backbone), **limit) as stats, torch.inference_mode():
        for text in ('chair', 'desk', 'desk', 'chair'):
            assert backbone.forward_text([text], device='cpu')['features'].item() == len(text)
        assert stats['entries'] == 1 and stats['bytes'] == 4
        assert backbone.calls == 3


def test_autocast_modes_do_not_share_features():
    backbone = Backbone()
    with cached_text_features(processor(backbone)) as stats, torch.inference_mode():
        backbone.forward_text(['chair'], device='cpu')
        with torch.autocast('cpu', dtype=torch.bfloat16):
            backbone.forward_text(['chair'], device='cpu')
            backbone.forward_text(['chair'], device='cpu')
        assert stats['entries'] == 2 and backbone.calls == 2


def test_training_gradient_and_geometric_calls_bypass_cache():
    backbone = Backbone()
    with cached_text_features(processor(backbone)) as stats:
        for _ in range(2):
            backbone.forward_text(['chair'], device='cpu')
        with torch.inference_mode():
            for _ in range(2):
                backbone.forward_text(['chair'], input_boxes=torch.ones(1), device='cpu')
                backbone.forward_text(['chair'], additional_text=['desk'], device='cpu')
            backbone.training = True
            for _ in range(2):
                backbone.forward_text(['chair'], device='cpu')
        assert stats['entries'] == 0 and backbone.calls == 8


def test_original_instance_override_is_restored_on_failure():
    backbone = Backbone()
    original = backbone.forward_text
    backbone.forward_text = original
    with pytest.raises(RuntimeError, match='worker failure'):
        with cached_text_features(processor(backbone)), torch.inference_mode():
            backbone.forward_text(['chair'], device='cpu')
            raise RuntimeError('worker failure')
    assert backbone.forward_text is original
    with cached_text_features(processor(backbone)), torch.inference_mode():
        backbone.forward_text(['chair'], device='cpu')
    assert backbone.calls == 2


@pytest.mark.parametrize('modern', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_checkpoint_load_preserves_copy_cast_and_legacy_formats(tmp_path, modern, dtype):
    from pose_pipeline.sam3_mapping import _load_image_checkpoint
    path = tmp_path / 'weights.pt'
    weights = torch.tensor([[1.25, -2.5]], dtype=dtype)
    torch.save({'model': {'detector.weight': weights,
                         'detector.unused': torch.ones(1)}}, path,
               _use_new_zipfile_serialization=modern)
    model = torch.nn.Linear(2, 1, bias=False)
    audit = {}
    _load_image_checkpoint(model, path, audit)
    assert torch.equal(model.weight, weights.float())
    assert model.weight.dtype == torch.float32
    assert audit['checkpoint_assign'] == (modern and dtype == torch.float32)
    assert audit['unexpected_keys'] == ['unused']


def test_checkpoint_missing_weights_still_rejects_random_fallback(tmp_path):
    from pose_pipeline.sam3_mapping import _load_image_checkpoint
    path = tmp_path / 'partial.pt'
    torch.save({'detector.bias': torch.ones(1)}, path)
    with pytest.raises(RuntimeError, match='incomplete SAM 3'):
        _load_image_checkpoint(torch.nn.Linear(2, 1), path, {})

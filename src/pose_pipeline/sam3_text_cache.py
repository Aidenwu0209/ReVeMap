"""Bounded, worker-local reuse of image-independent SAM3 text features."""
from contextlib import contextmanager


@contextmanager
def cached_text_features(processor, *, max_entries=64, max_bytes=16 * 1024**2):
    """Reuse plain text encoding only during inference on this fixed model.

    Keep the provider's prompt/grounding path intact. Return private tensor
    copies so downstream in-place operations cannot contaminate later frames.
    Never cache geometric prompts, training calls, or across worker/model loads.
    """
    import torch
    backbone = processor.model.backbone
    original = backbone.forward_text
    had_override = 'forward_text' in vars(backbone)
    cache = {}
    stats = {'hits': 0, 'misses': 0, 'entries': 0, 'bytes': 0}

    def forward(captions, input_boxes=None, additional_text=None, device='cuda'):
        if (backbone.training or not torch.is_inference_mode_enabled()
                or input_boxes is not None or additional_text is not None
                or not isinstance(captions, (list, tuple))
                or not all(isinstance(text, str) for text in captions)):
            return original(captions, input_boxes=input_boxes,
                            additional_text=additional_text, device=device)
        target = torch.device(device)
        device_index = target.index
        if target.type == 'cuda' and device_index is None:
            device_index = torch.cuda.current_device()
        autocast = torch.is_autocast_enabled(target.type)
        key = (tuple(captions), target.type, device_index, autocast,
               torch.get_autocast_dtype(target.type) if autocast else None)
        if key in cache:
            stats['hits'] += 1
            return {name: value.clone() for name, value in cache[key].items()}
        stats['misses'] += 1
        result = original(captions, input_boxes=input_boxes,
                          additional_text=additional_text, device=device)
        if isinstance(result, dict) and all(isinstance(v, torch.Tensor) for v in result.values()):
            size = sum(v.numel() * v.element_size() for v in result.values())
            if len(cache) < max_entries and stats['bytes'] + size <= max_bytes:
                cache[key] = {name: value.clone() for name, value in result.items()}
                stats.update(entries=len(cache), bytes=stats['bytes'] + size)
        return result

    backbone.forward_text = forward
    try:
        yield stats
    finally:
        if had_override:
            backbone.forward_text = original
        else:
            del backbone.forward_text
        cache.clear()

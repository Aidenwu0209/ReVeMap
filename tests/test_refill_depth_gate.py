from types import SimpleNamespace

import numpy as np

from pose_pipeline import refill_depth_gate as gate


def test_requires_depth_support_on_both_sides_and_never_changes_anchors(monkeypatch):
    frames = [SimpleNamespace(frame_id=i, timestamp_us=i*100_000) for i in range(9)]
    manifest = SimpleNamespace(frames=frames, depth_scale=1000)
    graph = np.repeat(np.eye(4)[None, :, :], len(frames), axis=0)
    refill = graph.copy()
    refill[4, 2, 3] = 0.4
    depth = np.full((100, 128), 2.0, dtype=np.float32)
    def read(frame, scale):
        return depth, (90., 90., 64., 50.)
    monkeypatch.setattr(gate, "_read_depth", read)
    corrected, audit = gate.verify_refill(manifest, graph, refill, [0,8])
    assert audit["restored_ordinals"] == [4]
    np.testing.assert_array_equal(corrected, graph)
    assert refill[4, 2, 3] == 0.4

    # Foreground occlusion in the next view removes graph support there.
    monkeypatch.setattr(gate, "_read_depth",
                        lambda f, s: (depth*0.5, (90., 90., 64., 50.)) if f.frame_id == 5 else read(f,s))
    kept, audit = gate.verify_refill(manifest, graph, refill, [0,8])
    assert audit["restored_ordinals"] == []
    np.testing.assert_array_equal(kept, refill)


def test_timestamp_gap_cannot_supply_missing_witness(monkeypatch):
    frames = [SimpleNamespace(frame_id=i, timestamp_us=t) for i,t in enumerate((0,1,4_000_000))]
    graph = np.repeat(np.eye(4)[None, :, :], len(frames), axis=0)
    refill = graph.copy()
    refill[1, 0, 3] = 1
    def no_read(*args):
        raise AssertionError("no pair needs scoring without two witnesses")
    monkeypatch.setattr(gate, "_read_depth", no_read)
    kept, audit = gate.verify_refill(SimpleNamespace(frames=frames, depth_scale=1000),graph,refill,[0,2])
    assert not audit["restored_ordinals"]
    np.testing.assert_array_equal(kept, refill)

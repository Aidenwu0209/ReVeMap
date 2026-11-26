
from pose_pipeline.semantic_runtime import naming_session
from pose_pipeline.semantic_runtime.common import LIVE_NAMERS, read, sha, write
from pose_pipeline.semantic_runtime.worker import name_tasks


def test_switch_between_crops_releases_old_model_and_records_actual_backend(tmp_path, monkeypatch):
    original, full_gpu = LIVE_NAMERS
    session = tmp_path / "session.json"
    image = tmp_path / "crop.png"
    image.write_bytes(b"fixed crop")
    crop = {"file": str(image), "sha256": sha(image)}
    order = [original, full_gpu, original]
    loads, resident, used = [], set(), []
    write(session, {"options": {"vlm": original}})
    monkeypatch.setenv("REVEMAP_NAMING_SESSION", str(session))

    class Namer:
        def __init__(self, model, config):
            assert not resident
            resident.add(model)
            loads.append(model)
            self.audit = {"id": model}
            self.model = model

        def infer(self, path):
            used.append(self.model)
            if len(used) < len(order):
                write(session, {"options": {"vlm": order[len(used)]}})
            return {"label": "chair", "valid": True, "executed": True, "request_seconds": 0.1}

        def close(self):
            resident.remove(self.model)

    monkeypatch.setattr(naming_session, "create_namer", Namer)
    output = tmp_path / "run"
    name_tasks([{"task_id": 0, "crops": [crop] * 3}], original, {}, output)
    assert not resident
    assert loads == used == order
    assert [r["model_id"] for r in read(output / "RECORDS.json")] == order
    assert [read(output / r["audit"])["id"] for r in read(output / "MODEL_LOADS.json")] == order

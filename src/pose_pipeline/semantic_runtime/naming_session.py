"""Select a local naming model between crops, keeping one backend loaded."""
import os
from pathlib import Path

from .common import LIVE_NAMERS, read, write
from .vlm import create_namer


class NamingSession:
    def __init__(self, initial, config, output):
        self.initial = initial
        self.config = config
        self.output = Path(output)
        self.session_file = os.environ.get("REVEMAP_NAMING_SESSION") if initial in LIVE_NAMERS else None
        self.model = None
        self.model_id = None
        self.loads = []

    def select(self):
        selected = self.initial
        if self.session_file:
            selected = read(self.session_file)["options"]["vlm"]
            if selected not in LIVE_NAMERS:
                raise ValueError("Unexpected live naming profile")
        if selected != self.model_id:
            self.close()
            folder = self.output if not self.loads else self.output / f"load_{len(self.loads):04d}"
            folder.mkdir(exist_ok=True)
            settings = {**self.config.get("models", {}).get(selected, {}), "log_dir": str(folder)}
            self.model = create_namer(selected, settings)
            self.model_id = selected
            write(folder / "MODEL.json", self.model.audit)
            self.loads.append({"model": selected, "audit": str((folder / "MODEL.json").relative_to(self.output))})
            if self.session_file:
                write(self.output / "MODEL_LOADS.json", self.loads)
        return self.model_id

    def infer(self, path):
        return {**self.model.infer(path), "model_id": self.model_id}

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
            self.model_id = None

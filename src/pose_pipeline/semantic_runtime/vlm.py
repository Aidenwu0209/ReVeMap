"""One image-naming contract for the actually tested model families.

Heavy runtimes are loaded lazily in a separate interpreter. Local inference uses
explicit checkpoint directories; APIs are only contacted when selected.
"""
from __future__ import annotations

import base64
import importlib
import json
import math
import mimetypes
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import types
from urllib.parse import urlparse

from .common import PROMPT, model_spec, parse_label, read, sha


def verify_weights(weights, spec):
    weights = Path(weights).resolve(strict=True)
    receipt = read(weights / "DOWNLOAD_COMPLETE.json")
    if receipt.get("revision") != spec["revision"] or receipt.get("repo") != spec["repo"]:
        raise ValueError("checkpoint receipt does not match the selected model revision")
    files = receipt.get("files", [])
    if not files:
        raise ValueError("empty checkpoint receipt")
    for row in files:
        path = (weights / row["file"]).resolve(strict=True)
        if not path.is_relative_to(weights) or sha(path) != row["sha256"]:
            raise ValueError("checkpoint file digest mismatch: " + row["file"])
    for name, digest in spec.get("reviewed_code_sha256", {}).items():
        if sha(weights / name) != digest:
            raise ValueError("reviewed model code changed: " + name)
    return weights, {"repo": spec["repo"], "revision": spec["revision"],
                     "receipt_sha256": sha(weights / "DOWNLOAD_COMPLETE.json"),
                     "verified_files": len(files)}


class Namer:
    audit = None

    def infer(self, path):
        start = time.perf_counter()
        answer, details = self._infer(Path(path))
        label, valid = parse_label(answer)
        return {"raw_response": answer, "label": label, "valid": valid,
                "request_seconds": time.perf_counter() - start, **details}

    def close(self):
        pass


class DisabledNamer(Namer):
    audit = {"model": "none", "weights_loaded": False}

    def _infer(self, path):
        return "unknown", {"executed": False}


class TransformersNamer(Namer):
    def __init__(self, spec, config):
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        self.spec, self.torch = spec, torch
        self.prompt = PROMPT
        self.weights, receipt = verify_weights(config["weights"], spec)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; local VLM does not silently fall back to CPU")
        torch.manual_seed(42)
        np.random.seed(42)
        torch.set_num_threads(int(config.get("threads", 4)))
        self.dtype = torch.float16 if spec["quant"] == "awq" else torch.bfloat16
        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        kwargs = {dtype_key: self.dtype, "device_map": {"": 0}, "local_files_only": True,
                  "trust_remote_code": False, "output_loading_info": True, "attn_implementation": "sdpa"}
        if spec["quant"] == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        if spec.get("preserve_tied_head"):
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(str(self.weights), local_files_only=True)
            if not cfg.tie_word_embeddings:
                raise ValueError("AWQ adapter requires the recorded tied output head")
            cfg.quantization_config["modules_to_not_convert"] = sorted(set(
                cfg.quantization_config.get("modules_to_not_convert", []) + ["lm_head"]))
            kwargs["config"] = cfg
        tick = time.perf_counter()
        if spec["kind"] == "mage_custom":
            self.model, loading, self.processor = self._load_mage(kwargs)
        elif spec["kind"] == "minicpm_custom":
            from transformers import AutoModel, AutoTokenizer
            kwargs["trust_remote_code"] = True
            self.model, loading = AutoModel.from_pretrained(str(self.weights), **kwargs)
            self.processor = AutoTokenizer.from_pretrained(
                str(self.weights), local_files_only=True, trust_remote_code=True)
        else:
            self.model, loading = AutoModelForImageTextToText.from_pretrained(str(self.weights), **kwargs)
            proc = {k: spec[k] for k in ("min_pixels", "max_pixels") if k in spec}
            self.processor = AutoProcessor.from_pretrained(
                str(self.weights), local_files_only=True, trust_remote_code=False, **proc)
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError("incomplete VLM checkpoint load: " + str(loading))
        if spec.get("preserve_tied_head") and (
                self.model.get_output_embeddings().weight.data_ptr() !=
                self.model.get_input_embeddings().weight.data_ptr()):
            raise RuntimeError("AWQ tied output head was not preserved")
        self.model.eval()
        torch.cuda.synchronize()
        self.audit = {**receipt, "id": spec["id"], "kind": spec["kind"],
                      "quant": spec["quant"], "load_seconds": time.perf_counter() - tick,
                      "torch": torch.__version__, "transformers": transformers.__version__,
                      "model_class": type(self.model).__name__, "processor_class": type(self.processor).__name__,
                      "loading_info": {k: list(v) if isinstance(v, (set, tuple)) else v for k, v in loading.items()},
                      "parameter_devices": sorted({str(p.device) for p in self.model.parameters()}),
                      "gpu": torch.cuda.get_device_name(), "static_images_only": True,
                      "requested_thinking_disabled": True, "thinking_disabled_verified": False}

    def _load_mage(self, kwargs):
        from transformers import PreTrainedModel
        # The official override cannot return loading_info. Invoke the inherited
        # loader on the same reviewed class; streaming/codec paths stay inactive.
        package_name = "mage_checkpoint_" + self.spec["revision"][:12]
        package = types.ModuleType(package_name)
        package.__path__ = [str(self.weights)]
        sys.modules[package_name] = package
        cfg = importlib.import_module(package_name + ".configuration_mage_vl")
        mod = importlib.import_module(package_name + ".modeling_mage_vl")
        proc = importlib.import_module(package_name + ".processing_mage_vl")
        config = cfg.MageVLConfig.from_pretrained(str(self.weights), local_files_only=True)
        kwargs = dict(kwargs)
        kwargs.pop("trust_remote_code", None)
        model, info = PreTrainedModel.from_pretrained.__func__(
            mod.MageVLForConditionalGeneration, str(self.weights), config=config, **kwargs)
        model.model._streammind_model_path = str(self.weights)
        # The official builder drops trust_remote_code before AutoTokenizer.
        # Reuse the already loaded, reviewed config to avoid another [y/N] lookup.
        processor = proc.MageVLProcessor.from_pretrained(
            str(self.weights), local_files_only=True, config=config, min_pixels=self.spec["min_pixels"],
            max_pixels=self.spec["max_pixels"])
        return model, info, processor

    def _infer(self, path):
        from PIL import Image
        torch, spec = self.torch, self.spec
        with Image.open(path) as source:
            image = source.convert("RGB")
        source_size = list(image.size)
        minimum = spec.get("min_side", 0)
        if min(image.size) < minimum:
            scale = minimum / min(image.size)
            image = image.resize(tuple(max(minimum, math.ceil(x * scale)) for x in image.size),
                                 Image.Resampling.BICUBIC)
        if spec["kind"] == "minicpm_custom":
            with torch.inference_mode():
                answer = self.model.chat(image=None, msgs=[{"role": "user", "content": [image, PROMPT]}],
                                         tokenizer=self.processor, sampling=False, max_new_tokens=24, max_slice_nums=1)
            torch.cuda.synchronize()
            return str(answer), {"executed": True, "output_tokens": None}
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        extra = {"enable_thinking": False}
        gen = {"do_sample": False, "max_new_tokens": spec["max_new_tokens"], "use_cache": True}
        if spec["model"] == "minicpmv46":
            extra.update(downsample_mode=spec["downsample_mode"], max_slice_nums=1)
            gen["downsample_mode"] = spec["downsample_mode"]
        image_audit = {}
        if spec["kind"] == "mage_custom":
            # Mage's apply_chat_template delegates to its text tokenizer.
            # Its image processor must be invoked separately to encode pixels.
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **extra)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            required = ("pixel_values", "image_grid_thw", "patch_positions")
            if any(key not in inputs or inputs[key].numel() == 0 for key in required):
                raise RuntimeError("Mage image encoding is missing; refusing text-only naming")
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            actual_tokens = int((inputs["input_ids"] == image_token_id).sum().item())
            expected_tokens = int(inputs["image_grid_thw"].prod(-1).sum().item()) // self.processor.spatial_merge_size**2
            if actual_tokens != expected_tokens or expected_tokens <= 0:
                raise RuntimeError("Mage image token count does not match encoded image grid")
            image_audit = {"image_inputs_verified": True, "image_tokens": actual_tokens,
                           "image_tensor_shapes": {key: list(inputs[key].shape) for key in required}}
        else:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", **extra).to("cuda")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                inputs[key] = value.to(self.dtype)
        count = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen)
        torch.cuda.synchronize()
        answer = self.processor.batch_decode(output[:, count:], skip_special_tokens=True)[0]
        return answer, {"executed": True, "output_tokens": int(output.shape[-1] - count),
                        **image_audit,
                        "source_image_size": source_size, "adapter_image_size": list(image.size),
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}

    def close(self):
        del self.model, self.processor
        self.torch.cuda.empty_cache()


def _image_url(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()


class APINamer(Namer):
    def __init__(self, spec, config):
        self.spec = spec
        self.endpoint = config.get("endpoint", spec["endpoint"])
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query:
            raise ValueError("API endpoint must be HTTPS without embedded credentials")
        self.token = os.environ.get(config.get("token_env", spec["token_env"]))
        if not self.token:
            raise ValueError("missing API token environment variable")
        self.audit = {"id": spec["id"], "endpoint": self.endpoint, "role": spec["role"]}

    def _infer(self, path):
        import requests
        payload = {"model": self.spec["model"], "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_url(path)}},
            {"type": "text", "text": PROMPT}]}], "temperature": 0,
            "max_tokens": self.spec["max_new_tokens"], "thinking": {"type": "disabled"},
            "stream": False}
        # DeepSeek's reasoning_effort would re-enable thinking. Keep the GLM
        # request unchanged while matching the separately verified API payload.
        effort = self.spec.get("reasoning_effort", "low")
        if effort is not None:
            payload["reasoning_effort"] = effort
        if "image_detail" in self.spec:
            payload["messages"][0]["content"][0]["image_url"]["detail"] = self.spec["image_detail"]
        response = requests.post(self.endpoint, headers={"Authorization": "Bearer " + self.token},
                                 json=payload, timeout=(15, 60), allow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError(f"API HTTP {response.status_code}; no local-model fallback")
        body = response.json()
        choice = body["choices"][0]
        answer = choice["message"].get("content") or ""
        # Do not persist response headers, signed URLs, tokens, or full request bodies.
        return answer.replace(self.token, "[redacted]"), {"executed": True, "usage": body.get("usage"),
                                                           "response_model": body.get("model"),
                                                           "system_fingerprint": body.get("system_fingerprint"),
                                                           "finish_reason": choice.get("finish_reason"),
                                                           "reasoning_content_present": bool(choice["message"].get("reasoning_content"))}


class OCRNamer(APINamer):
    """OCR is kept testable but never presented as a 3D object-name predictor."""
    def infer(self, path):
        import requests
        start = time.perf_counter()
        options = {"useDocOrientationClassify": False, "useDocUnwarping": False,
                   "useLayoutDetection": True, "useChartRecognition": False,
                   "useSealRecognition": True, "useOcrForImageBlock": True, "temperature": 0, "topP": 1}
        auth = {"Authorization": "bearer " + self.token}
        with Path(path).open("rb") as stream:
            response = requests.post(self.endpoint, headers=auth, data={
                "model": self.spec["model"], "optionalPayload": json.dumps(options)},
                files={"file": (Path(path).name, stream, mimetypes.guess_type(str(path))[0] or "image/png")},
                timeout=(15, 60), allow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError(f"OCR submission HTTP {response.status_code}")
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError("OCR submission rejected")
        job = body["data"]["jobId"]
        if not isinstance(job, str) or not job or any(x in job for x in "/?#"):
            raise ValueError("invalid OCR job ID")
        interval = 3.
        while time.perf_counter() - start < self.spec.get("poll_timeout", 600):
            poll = requests.get(self.endpoint + "/" + job, headers=auth,
                                timeout=(15, 30), allow_redirects=False)
            if poll.status_code != 200:
                raise RuntimeError(f"OCR polling HTTP {poll.status_code}")
            result = poll.json()
            if result.get("code") != 0:
                raise RuntimeError("OCR polling rejected")
            data = result.get("data", {})
            if data.get("state") == "failed":
                raise RuntimeError("OCR service job failed")
            if data.get("state") == "done":
                url = data.get("resultJsonUrl") or data.get("resultUrl", {}).get("jsonUrl")
                if not isinstance(url, str) or urlparse(url).scheme != "https":
                    raise RuntimeError("OCR did not return an HTTPS result URL")
                # The authorized service supplies this URL. Never forward its token.
                response = requests.get(url, timeout=(15, 60))
                response.raise_for_status()
                pages = [json.loads(line) for line in response.text.splitlines() if line.strip()]
                texts = []
                for page in pages:
                    for item in page.get("result", {}).get("layoutParsingResults", []):
                        texts += [x.get("block_content", "") for x in item.get("prunedResult", {}).get("parsing_res_list", [])]
                return {"ocr_text": "\n".join(texts).replace(self.token, "[redacted]"),
                        "label": "unknown", "valid": False, "role": "ocr_evidence_only",
                        "executed": True, "request_seconds": time.perf_counter() - start}
            time.sleep(interval)
            interval = min(15., interval * 1.5)
        raise TimeoutError("OCR observation window expired; job may still be queued")


class LlamaNamer(Namer):
    def __init__(self, spec, config):
        import requests
        self.spec, self.child, self.log = spec, None, None
        weights, receipt = verify_weights(config["weights"], spec)
        binary = Path(config["llama_server"]).resolve(strict=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}"
        cmd = [str(binary), "-m", str(weights / spec["gguf"]), "--mmproj", str(weights / spec["mmproj"]),
               "--host", "127.0.0.1", "--port", str(port), "-ngl", "999", "--mmproj-offload", "--fit", "off",
               "-c", "2048", "-np", "1", "-t", "4", "-tb", "4", "-b", "256", "-ub", "256",
               "--flash-attn", "on", "--image-max-tokens", str(spec["image_tokens"]),
               "--no-cache-prompt", "--cache-ram", "0", "--no-cache-idle-slots",
               "--reasoning", "off", "--seed", "42"]
        if "image_min_tokens" in spec:
            cmd += ["--image-min-tokens", str(spec["image_min_tokens"])]
        log_path = Path(config["log_dir"]) / "llama-server.log"
        self.log = log_path.open("x")
        try:
            tick = time.monotonic()
            self.child = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
            while time.monotonic() - tick < 180:
                if self.child.poll() is not None:
                    raise RuntimeError("llama-server exited; inspect its log")
                try:
                    if requests.get(self.endpoint + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError("llama-server readiness timeout")
            self.audit = {**receipt, "id": spec["id"], "backend": "llama.cpp CUDA", "command": cmd,
                          "binary_sha256": sha(binary), "load_seconds": time.monotonic() - tick}
        except BaseException:
            self.close()
            raise

    def _infer(self, path):
        import requests
        payload = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_url(path)}},
            {"type": "text", "text": PROMPT}]}], "temperature": 0, "top_k": 1, "seed": 42,
            "max_tokens": 24, "cache_prompt": False, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
        response = requests.post(self.endpoint + "/v1/chat/completions", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"].get("content") or "", {
            "executed": True, "usage": data.get("usage"), "server_timings": data.get("timings")}

    def close(self):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        if self.log is not None:
            self.log.close()


def create_namer(model_id, config=None):
    spec, config = model_spec(model_id), config or {}
    cls = {"none": DisabledNamer, "api": APINamer, "ocr": OCRNamer, "llama": LlamaNamer}.get(
        spec["kind"], TransformersNamer)
    return cls() if cls is DisabledNamer else cls(spec, config)

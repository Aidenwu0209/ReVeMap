# ReVeMap capture GUI

The browser GUI supports Orbbec capture or registered RGB-D replay, live geometry
preview, and semantic processing after recording stops. It invokes the current
repository's `run-sam3` path and optional unknown-point refinement. It does not
automatically generate the frozen T1/P2 evidence bundle.

## Start

Prepare the external environments and runtime described in [SETUP.md](../SETUP.md).
Run on the machine connected to the camera and GPU:

```bash
revemap gui \
  --runtime /absolute/path/runtime.local.json \
  --provider-root /absolute/path/DROID-W \
  --cpu-python /absolute/path/cpu-env/bin/python \
  --gpu-python /absolute/path/droid-env/bin/python \
  --capture-python /absolute/path/orbbec-env/bin/python \
  --output /absolute/path/ReVeMap-scans
```

The capture interpreter needs the Orbbec SDK, OpenCV and NumPy. The other paths
must match the environments in the runtime JSON. Keep virtual-environment
interpreter paths rather than resolving their symlinks to system Python.

Open `http://127.0.0.1:8765`. The server only listens on loopback. For a remote
GPU host, forward that port through SSH using your own hostname/account. Use
`--port 8767` for a different port or `--no-browser` on a headless host.

The source launcher is also available:

```bash
bash scripts/run_revemap_gui.sh --help
```

Set `REVEMAP_CPU_PYTHON` if the CPU interpreter is not the active `python3`.
Pass the same arguments as `revemap gui`; there are no hardcoded host paths.

## Capture and replay

1. Choose the schedule and a configured naming model.
2. Start capture and move the camera slowly while watching RGB/depth preview.
3. Stop recording to run geometry, SAM3, fusion and naming; optional refinement
   then adds supported unknown labels.
4. Inspect semantic, instance and original-color views; download PLY and poses.

Refinement has its own Qwen NF4/SAM3 stage. To disable all VLM inference, select
`none` and turn off refinement. Cancel preserves recorded input rather than
claiming a completed map.

For registered recorded data, add:

```bash
--replay /absolute/path/registered-manifest.json --fps 30 --max-frames 120
```

The limit is a replay option. Each scan receives a new `scan_*` directory.
`pipeline/GUI_RESULT.json` identifies the final map, while `mapping.log` and
`pipeline/*.log` retain stage diagnostics. Preview subsampling does not change
the point count of the exported PLY.

## Validation boundary

The repository extraction checks CPU logic, packaging, routes and entry points.
A successful help command or rendered interface does not validate camera data,
GPU model loading, full recording/export or cancellation with real hardware.

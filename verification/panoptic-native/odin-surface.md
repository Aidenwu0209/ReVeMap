# ODIN multi-view instance observations

These optional tools export cross-view instance queries on a reconstructed RGB-D surface and use them to split, join or extend existing instances. The original mapping pipeline supplies the colored triangle mesh, map point cloud and estimated trajectory. The association step reads the PTv3 surface probabilities and its refined panoptic map in the same map point order.

## Runtime and inputs

Use the author's ScanNet20 Swin checkout and `scannet_swin_50.0_64k_6k.pth`. The verified checkpoint has SHA-256 `fab498c19378a4ea97c1aab320451ce5112bb15bbccba0cd200c7df74d3202d1`. The author's published 50.0 measures instance AP. ReVeMap evaluates complete-ground-truth-domain PQ independently. [ODIN source and weights](https://github.com/ayushjain1144/odin)

- The GPU exporter uses Python, PyTorch with CUDA, Detectron2's source tree, `pytorch3d`, `torch_scatter`, the author's `libs/pointops2` CUDA extension, OpenCV, Pillow, `plyfile` and SciPy. The real replay used PyTorch 2.7.1 and CUDA 12.8. The exporter creates a temporary inference-only import tree and retains the author's model layers and PyTorch deformable-attention implementation. Its receipt records the source fingerprint, import adjustments and strictly loaded state dictionary.
- The sensor input is an aligned, JPEG-color and zlib-ushort-depth ScanNet v4 `.sens` recording. The trajectory is a ReVeMap JSON with `matrix_convention="T_world_camera_m"` and one chronologically ordered pose row per source frame. Both view policies use the recording's depth intrinsics and raw measured depth.
- The geometric preparation requires a ScanNet-style `segmentator` installation, NumPy, SciPy and `plyfile`. This step can run in a separate environment from ODIN. The final association is CPU-only.

## Run

Set the shell variables in these commands to the paths of the actual inputs. `RECON` contains the target `mesh.ply`, `map.ply` and `trajectory.json`. If using a reintegrated mesh with the original room axes, set `ANCHOR` to the original reconstructed mesh and estimated trajectory. The output directory may be anywhere writable.

```sh
python tools/prepare_odin_geometry.py \
  --mesh "$RECON/mesh.ply" --trajectory "$RECON/trajectory.json" \
  --anchor-mesh "$ANCHOR/mesh.ply" --anchor-trajectory "$ANCHOR/trajectory.json" \
  --output "$OUT/geometry"

python tools/export_odin_queries.py \
  --sens "$SENS" --trajectory "$RECON/trajectory.json" \
  --mesh "$RECON/mesh.ply" --map "$RECON/map.ply" \
  --odin-root "$ODIN" --detectron2-root "$DETECTRON2" --weights "$WEIGHTS" \
  --superpoints "$OUT/geometry/groups.npz" --upright "$OUT/geometry/geometry.json" \
  --output "$OUT/queries"

python tools/associate_odin_instances.py \
  --map "$RECON/map.ply" --base "$PTV3/extend.npz" \
  --point-evidence "$POINT_EVIDENCE" \
  --view19 "$OUT/queries/view19/queries.npz" --cap64 "$OUT/queries/cap64/queries.npz" \
  --output "$OUT/association"
```

When the target mesh already defines the desired room axes, omit both anchor arguments. The exporter also accepts an existing upstream result JSON with a `rotation` or `rotation_xyz_to_upright` matrix. Without `--upright`, it estimates the room axes from the supplied target mesh and estimated camera trajectory. Without `--superpoints`, it computes the target geometric groups in its own environment. Add repeatable `--python-path` options for separately installed model dependencies. `--lock /path/to/shared.lock` coordinates CUDA use with other processes that use the same advisory Linux file lock.

## Formats

| File | Arrays |
| --- | --- |
| Both `queries.npz` files | `semantic[Nmesh]` in canonical ScanNet20 order, `classes[100]` and `scores[100]` for the author's selected instance/class combinations, `masks[100,ceil(Nmesh/8)]` containing binary mesh masks, `map_to_mesh[Nmap]` int32, and scalar `mesh_count` |
| Input `extend.npz` | Canonical `semantic[Nmap]` and integer `instance[Nmap]` |
| Input point evidence | `proba[Nmesh,20]` quantized to uint8 over 0–255, `nearest[Nmap]` mesh indices, and `distance[Nmap]` in metres |
| Output `combined.npz` | `semantic[Nmap]` uint8 and `instance[Nmap]` int32, in exactly the input map order |

```python
import numpy as np

with np.load(query_file) as q:
    mesh_mask = np.unpackbits(q["masks"][query_index], count=int(q["mesh_count"]))
    map_mask = mesh_mask[q["map_to_mesh"]].astype(bool)
```

The 19-view pass selects camera poses with complementary position and viewing direction. The 64-view pass starts from every twentieth valid-indexed raw frame and samples at most 64 evenly over time. Both use 320×240 RGB-D. The author's wall and floor classes 19 and 20 map to ReVeMap classes 0 and 1. Each instance candidate carries its category probability multiplied by the mean confidence inside its predicted mask.

The association decodes candidates above 0.55. Two camera samplings must agree before an existing same-class instance is split or multiple instances are joined. An extension additionally requires PTv3 point support and proximity to the observed mesh. Each command seals its output with SHA-256 hashes and records the inputs. Reusing an output location with a different signature raises an error and preserves the existing result.

The single-room replay is recorded in `odin-scene0070-replay.json`. On the nine official-training scans, the frozen rule changed mean PQ from 60.153% to 61.013% using the same mixed-graph coordinates, with one additional true instance and three fewer false positives.

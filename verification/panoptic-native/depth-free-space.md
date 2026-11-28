# 用两条估计轨迹检查补点是否位于空闲空间

这个过滤器输出与补点逐行对应的布尔数组 `keep`。它读取原始 ScanNet v4 深度、联合位姿图修正后的主轨，以及 ORB 半幅重融使用的辅助轨，两条轨迹均以米表示、采用同一个原生 DROID 世界系。原始主图和已分配的实例编号由合并步骤保管。[过滤器](../../tools/filter_free_space_points.py) · [辅助轨迹构建](orb-half-surface.md)

## 从估计轨迹运行

下例中的 `NATIVE_A` 和 `NATIVE_TRAJECTORY` 来自同次 ReVeMap 运行。`PRIMARY_LABELS` 是原生 A 点序的边界精修标签 PLY。MASt3R 与 ORB 的配置、编译和权重见各自的入口说明。[联合位姿图](joint-geometry.md) · [ORB 半幅重融](orb-half-surface.md)

```bash
python tools/refine_scannet_loop_geometry.py measure \
  --sensor "$SENS" --trajectory "$NATIVE_TRAJECTORY" \
  --checkpoint "$MAST3R_CHECKPOINT" --output "$LOOP_DIR"
python tools/refine_scannet_loop_geometry.py apply \
  --sensor "$SENS" --trajectory "$NATIVE_TRAJECTORY" --map "$NATIVE_A" \
  --labeled "$PRIMARY_LABELS" --edges "$LOOP_DIR/edges.json" \
  --save-trajectory --output "$PRIMARY"
python tools/build_orb_half_surface.py \
  --sensor "$SENS" --trajectory "$NATIVE_TRAJECTORY" --native-map "$NATIVE_A" \
  --orb-runner build/orb_rgbd/revemap_orb_rgbd \
  --vocabulary "$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt" \
  --expert-python "$ONEFORMER_PYTHON" --config "$ONEFORMER_SCAN20_CONFIG" \
  --checkpoint "$ONEFORMER_SCAN20_CHECKPOINT" --output "$SECONDARY"
```

`--save-trajectory` 将图节点相对原轨的旋转修正按帧号球面插值、平移修正按帧号线性插值，再把 0.6 比例的修正施加于各帧原估计。文件 `$PRIMARY/trajectory.json` 与该次输出的主图共享坐标系。`$SECONDARY/trajectory.json` 是实际重融所用的 0.5 比例 DROID–ORB 轨迹。两条完整轨迹的帧号均与原始传感器记录逐帧对应。

从原有补面命令生成的联合 PLY 中，按主图点数截取连续尾部，即可独立导出补点掩码。已有同序 `xyz` 补点 NPZ 的调用方可直接运行最后一条命令。[联合 PLY 输出](complementary-surfaces.md)

```bash
python tools/fuse_complementary_surfaces.py \
  --primary "$PRIMARY/map_labeled.ply" --secondary "$SECONDARY/single/map_labeled.ply" \
  --primary-in-secondary "$SECONDARY/primary_in_secondary.ply" --output "$RAW_UNION"
python - "$PRIMARY/map_labeled.ply" "$RAW_UNION/map_labeled.ply" "$ADDITIONS_NPZ" <<'PY'
import sys
import numpy as np
from plyfile import PlyData
primary, union, output = sys.argv[1:]
start = PlyData.read(primary)["vertex"].count
tail = PlyData.read(union)["vertex"].data[start:]
np.savez_compressed(output, xyz=np.column_stack([tail[k] for k in "xyz"]))
PY
python tools/filter_free_space_points.py \
  --sensor "$SENS" --primary-trajectory "$PRIMARY/trajectory.json" \
  --secondary-trajectory "$SECONDARY/trajectory.json" --additions "$ADDITIONS_NPZ" \
  --output "$KEEP_NPZ"
```

库函数 `free_space_keep(sensor, primary_poses, secondary_poses, additions)` 接受两组形状为 `F × 4 × 4` 的相机到世界矩阵和一组 `N × 3` 米制补点，返回长度为 `N` 的布尔数组。独立过滤时，将 `True` 对应的补点按原序接回全部主点。`NPZ` 输入仅需 NumPy；读取 PLY 另需 `plyfile`。

## 固定参数和实测

每个 4 cm 体素选原序中首个点代表该体素。分别沿两条估计轨迹投影到原始深度分辨率，每 30 帧取一次，并包含最后一帧；投影内参取自传感器头部的深度相机矩阵。候选深度须在 0.30–4.45 m 内。中心和上下左右五个像素均须位于 0.25–4.5 m，且最大深度差小于 8 cm。令 `z` 为预测深度、`r` 为中心像素实测深度减去 `z`。当 `|r| ≤ 0.035 + 0.004z` 时计一次表面支持，当 `r > 0.105 + 0.008z` 时计一次空闲空间观测。只有两条轨迹各自都得到至少两次空闲观测、且其次数超过表面支持时，才移除该体素内的补点。两组投影共享同一条深度录制。

在 9 条训练扫描上选定该规则后，冻结所有阈值。单独对 142 条既定验证扫描实施过滤，5 cm 地图精确率从 47.8372% 提高到 48.0382%，补点精确率从 18.8438% 提高到 19.1747%。覆盖率从 86.8204% 降到 86.6247%，PQ 从 50.5474% 降到 50.4993%。32 扫描 PQ 提高、37 持平、73 下降。这些数字采用逐扫描等权均值。

训练扫描 `scene0062_00` 中，用公开测量文件和原生地图重新求解得到 125 个图节点、523 条短边、4 条长边。730 个逐帧相机中心相对封存值最大偏差为 0.01791 mm，最大旋转差为 0.00052°。13,608 个补点中保留 13,596 个；公开重算的主轨和封存主轨均逐点复现冻结掩码。验证扫描 `scene0063_00` 用封存的两条估计轨迹和公开过滤器保留 223,707/225,154 个补点，布尔数组同样逐点一致。[训练场收据](depth-free-space-0062.json) · [验证场收据](depth-free-space-0063.json)

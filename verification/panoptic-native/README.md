# 原生地图上的 ScanNet20 全景精修

在既定 142 条验证扫描上，重融网格四次投票加联合位姿图得到平均 PQ 46.37%，mIoU 56.21%。[复现命令、逐扫描配对与适用范围](joint-geometry.md)

如需在重融表面上联合四个视角并补全实例边界，可接入 Point Transformer V3。[运行命令和逐点复核](ptv3-surface.md)

在 RGB-D 建图和 SAM3 实例精修结束后，可对同次采集的彩色 TSDF 网格运行 OneFormer3D。模型在网格上生成超点和新的实例边界，再将标签投回原有地图点。传入类别表后，ReVeMap 的多视角实例还可恢复漏掉的物体，并扩展两种预测共同支持的表面。

使用作者的 OneFormer3D、segmentator 和匹配版本的 CUDA、PyTorch、spconv、MMDetection3D 环境。安装参考[作者仓库](https://github.com/filaPro/oneformer3d)。脚本校验 ScanNet20 发布版权重的 SHA-256。[权重记录](checkpoint.json)

```bash
python tools/refine_native_panoptic.py \
  --mesh "$NATIVE_MESH" --map "$COMPLETED_MAP" \
  --trajectory "$ESTIMATED_TRAJECTORY" --native-classes "$MAP_CLASSES" \
  --config "$ONEFORMER_SCAN20_CONFIG" --checkpoint "$ONEFORMER_SCAN20_CHECKPOINT" \
  --output "$NEW_RESULT_DIR"
```

输入网格需带 RGB 和三角面，地图需带 `semantic_id`、`instance_id`。两者使用同一重建坐标系。轨迹使用 ReVeMap 的 `T_world_camera_m` 格式。输出到新目录，包含彩色点云、类别表、逐点标签和哈希收据。实例编号属于这份新地图。省略 `--native-classes` 可复现只使用三维模型的对照。

### 142 条验证扫描

项目已有的这 142 条扫描均属于 ScanNet 官方验证划分。对每条扫描使用首对有效相机位姿将重建对齐到参考系，在完整真值顶点上采用 5 cm 最近邻投影。每类实例按 IoU 严格大于 0.5 匹配，PQ 先对每场存在的类别取平均，再对扫描等权平均。三组使用相同原生点和估计轨迹。推理阶段只使用重建结果。

| 方案 | PQ（%） | mIoU（%） | 物体命中 | 物体误检 |
|---|---:|---:|---:|---:|
| ReVeMap 原主线 | 24.78 | 39.14 | 574 | 2,307 |
| 加三维专家 | 39.00 | 49.74 | 915 | 1,305 |
| 加三维专家并结合原实例扩边 | 39.48 | 50.38 | 931 | 1,363 |

联合输出相对主线找回 433 个物体、丢失 76 个，133 场 PQ 提高、2 场持平、7 场下降。相较只用三维专家，结合原实例增加 16 个净命中及 58 个误检。作者在 ScanNet 参考网格上得到的成绩使用不同输入。[逐扫描配对](validation142.csv) · [汇总](validation142.json)

### 用原始 RGB-D 修正地图坐标

在同一原生网格上增加两次水平旋转和一次 3.5 cm 顶点聚类采样，再对四份预测匹配实例和投票。将上面命令中的 `--native-classes "$MAP_CLASSES"` 替换为 `--vote-views` 即可生成投票地图。GPU 每次处理一个视图。

下一步沿用 DROID 的估计轨迹，固定第 0 帧。双向 RGB-D 里程计检验深度与颜色的一致性，SIFT 特征以深度和双向重投影误差提供另一组约束。两组修正方向一致的地图点会获得额外的特征位移。`--smooth` 在相邻区域间平滑修正量。坐标以外的逐点字段原样传递。CPU 运行环境需要 NumPy、SciPy、OpenCV、Open3D 和 plyfile。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python tools/refine_scannet_rgbd_geometry.py \
  --sensor "$SCENE_SENS" --trajectory "$ESTIMATED_TRAJECTORY" \
  --map "$VOTED_RESULT_DIR/map_labeled.ply" --smooth --output "$GEOMETRY_RESULT_DIR"
```

输入是与建图相同的 ScanNet v4 JPEG/zlib RGB-D 流和米制估计轨迹。读取器跳过传感器文件中的参考位姿。9 条官方训练扫描用于开发。下表沿用前述 142 条验证扫描和评分器。

| 逐点标签 | 坐标处理 | PQ（%） | mIoU（%） | 物体命中 | 物体误检 |
|---|---|---:|---:|---:|---:|
| 三维专家结合原实例扩边 | 原始 | 39.48 | 50.38 | 931 | 1,363 |
| 同上 | 双向 RGB-D | 42.95 | 53.19 | 1,001 | 1,326 |
| 同上 | 双向 RGB-D 和特征约束 | 43.23 | 53.43 | 1,007 | 1,318 |
| 同上 | 加局部平滑 | 43.43 | 53.53 | 1,005 | 1,312 |
| 四视图三维投票 | 原始 | 40.19 | 50.90 | 948 | 1,255 |
| 四视图三维投票 | 双向 RGB-D | 43.75 | 53.74 | 1,019 | 1,215 |
| 四视图三维投票 | 双向 RGB-D、特征约束和局部平滑 | **44.27** | **54.08** | **1,021** | **1,207** |

最后一组比原主线提高 19.49 个 PQ 百分点。2,116 个真值物体中，相对主线找回 535 个、丢失 88 个，131 条扫描提高、1 条持平、10 条下降。逐场先封存全部预测，再评分。独立命令在开发扫描 `scene0062_00` 的 94,618 个顶点逐点复现封存的坐标、类别和实例编号。[配对分数](geometry142.csv) · [汇总](geometry142.json) · [命令复现](packaged-vote-geometry-0062.json)

### 用修正后的相机轨迹重新融合表面

每隔 6 帧，将真实深度和颜色重新融合进 2 cm TSDF 网格。相机轨迹使用双向 RGB-D 估计的半幅修正，原地图点按同一幅度移动。随后让三维专家在新的彩色网格上重新划分实例，并用原有观测复核边界。两条命令分别输出重建和分割结果。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python tools/reintegrate_scannet_rgbd.py \
  --sensor "$SCENE_SENS" --trajectory "$ESTIMATED_TRAJECTORY" \
  --map "$COMPLETED_MAP" --output "$NEW_MESH_DIR"

python tools/refine_native_panoptic.py \
  --mesh "$NEW_MESH_DIR/mesh.ply" --map "$NEW_MESH_DIR/map.ply" \
  --trajectory "$NEW_MESH_DIR/trajectory.json" --native-classes "$MAP_CLASSES" \
  --config "$ONEFORMER_SCAN20_CONFIG" --checkpoint "$ONEFORMER_SCAN20_CHECKPOINT" \
  --output "$NEW_PANOPTIC_DIR"
```

142 条验证扫描上，重新融合并重新推理得到 **PQ 44.72%、mIoU 54.67%**。与使用相同半幅坐标的原网格实例扩边相比，PQ 从 42.95% 升至 44.72%，物体命中从 1,001 到 1,026，误检从 1,326 降至 1,165。相对原主线，新增命中 530 个、失去 78 个，135 条扫描提高、1 条持平、6 条下降。独立命令在训练扫描 `scene0062_00` 上重现全部 730 个估计位姿，重融网格、同序地图和逐点全景预测分别与封存文件的 SHA-256 一致。[逐扫描结果](reintegration142.csv) · [汇总](reintegration142.json) · [命令复现](packaged-reintegration-0062.json)

# 用独立三维语义预测扩展重融网格上的实例

对同一重融网格依次运行两条命令。第一步用原生物体观测复核四个视角的实例预测，然后进行掩码投票。第二步使用 Point Transformer V3 根据顶点的颜色、位置和三角面法向预测 ScanNet20 语义，在语义连续的相邻三角面上将有支持的点分给已有实例。

```bash
python tools/refine_native_panoptic.py \
  --mesh "$NEW_MESH_DIR/mesh.ply" --map "$NEW_MESH_DIR/map.ply" \
  --trajectory "$NEW_MESH_DIR/trajectory.json" --native-classes "$MAP_CLASSES" \
  --config "$ONEFORMER_SCAN20_CONFIG" --checkpoint "$ONEFORMER_SCAN20_CHECKPOINT" \
  --vote-views --output "$VOTED_OUTPUT"

python tools/extend_panoptic_surface.py \
  --mesh "$NEW_MESH_DIR/mesh.ply" --map "$NEW_MESH_DIR/map.ply" \
  --anchor "$VOTED_OUTPUT" --checkpoint "$PTV3_SCANNET_CHECKPOINT" \
  --output "$SURFACE_OUTPUT"
```

第一步使用 OneFormer3D 的运行环境。第二步使用 Point Transformer V3 的作者实现，实测环境为 PyTorch 2.11.0+cu128。验证使用的源码版本为 `4e277e6802eb3a6299fd71f707394d918b96f5fc`。模型权重选用作者发布的 ScanNet20 `model_best.pth`，训练配置中的训练集为 ScanNet train。脚本校验模型、网格、地图和第一阶段逐点输出的哈希。[模型实现](https://github.com/Pointcept/PointTransformerV3/tree/4e277e6802eb3a6299fd71f707394d918b96f5fc) · [权重和训练配置](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/scannet-semseg-pt-v3m1-0-base)

四个视角依次为原朝向、水平旋转 45°、水平旋转 90° 和 3.5 cm 简化网格。每个视角单独进行原生实例复核。第二步从收据读取第一步的重力旋转，沿用同一顶点顺序，输出 `map_labeled.ply`、`panoptic.npz`、`classes.json` 及哈希收据 `result.json`。

两条命令已分别在 94,618 点与 880,493 点的开发地图上对照研究实现。两个阶段输出的逐点类别、实例均完全一致，结果 PLY 的坐标和 RGB 与输入地图逐点一致。[scene0062_00 收据](scene0062_equivalence.json) · [scene0000_00 收据](scene0000_equivalence.json)

### 重融网格四视角的 142 条验证扫描

用同一份重融彩色网格、同一组实例观测和同一条地图点序，比较单视角与四视角。表中两项分别使用半幅双向 RGB-D 位姿修正和 0.6 幅混合位姿修正。全部预测封存后在完整真值顶点上按 5 cm 对齐距离和严格大于 0.5 的实例 IoU 评分，PQ 与 mIoU 均对扫描等权平均。

| 地图点坐标 | 实例预测 | PQ（%） | mIoU（%） | 物体命中 | 物体误检 |
|---|---|---:|---:|---:|---:|
| 双向 RGB-D × 0.5 | 单视角 | 44.72 | 54.67 | 1,026 | 1,165 |
| 双向 RGB-D × 0.5 | 四视角多数投票 | 45.83 | 55.42 | 1,056 | 1,145 |
| 混合位姿 × 0.6 | 单视角 | 45.21 | 55.46 | 1,037 | 1,188 |
| 混合位姿 × 0.6 | 四视角多数投票 | 46.37 | 56.21 | 1,065 | 1,170 |

混合位姿下，四视角相对单视角提高 1.16 个 PQ 百分点，95 场提高、2 场持平、45 场下降。以 134 个房间家族为单位重复抽样 4,000 次，配对差值的 95% 百分位区间为 0.54 至 1.81 个百分点。双向 RGB-D 坐标下为 97 场提高、3 场持平、42 场下降。[逐扫描分数](remesh-vote142.csv) · [汇总、配对区间和封存收据](remesh-vote142.json)

### 三维语义扩展和争议边界重分配

在相同的 142 场验证扫描上，按半幅双向 RGB-D 修正每六帧重新融合一次网格。先将独立 PTv3 预测的连通表面分给现有实例，再用同一份 PTv3 顶点概率和四次预测处理原有实例边界。表中两步在 9 场官方训练扫描上分别选定，使用全部验证预测封存后的完整真值评分。

| 地图点坐标 | 处理 | PQ（%） | mIoU（%） | 物体命中 | 物体误检 |
|---|---|---:|---:|---:|---:|
| 双向 RGB-D × 0.5 | 四视角投票 | 45.835 | 55.424 | 1,056 | 1,145 |
| 双向 RGB-D × 0.5 | 加 PTv3 扩展 | 46.627 | 56.648 | 1,077 | 1,133 |
| 混合位姿 × 0.6 | 四视角投票 | 46.373 | 56.209 | 1,065 | 1,170 |
| 混合位姿 × 0.6 | 加 PTv3 扩展 | 47.325 | 57.494 | 1,090 | 1,149 |
| 混合位姿 × 0.6 | 扩展后重分配争议边界 | **47.595** | **57.669** | **1,094** | **1,136** |

混合位姿下，PTv3 扩展令 105 场提高、4 场持平、33 场下降。配对 PQ 增量为 0.952 个百分点，95% 房间家族重采样区间为 0.186 至 1.725。争议边界重分配相对扩展单步又提高 0.270 个百分点，区间为 0.066 至 0.512。87 场提高、11 场持平、44 场下降，新增 7 个真值物体匹配，失去 3 个旧匹配。另一个在训练场选定的整实例剔除方法得到 47.169%，物体命中从 1,090 降至 1,081，误检从 1,149 降至 1,079。[扩展逐扫描数据](ptv3-surface142.csv) · [扩展汇总](ptv3-surface142.json) · [边界逐扫描数据](boundary142.csv) · [边界汇总及封存哈希](boundary142.json)

四视角对原实例类别的支持最多为两票、与网格距离小于 8 厘米、PTv3 预测类别不同且量化置信度大于 215/255 的点进入边界复核。对墙面或地面，只有当一个实例的争议点最多占 60% 时才允许释放。对物体类别，模型在 12 厘米内寻找已有同类实例的高置信度点，并继承最近点的实例编号。同一类别须有至少 30 个置信度大于 175/255 的候选点。

~~~bash
python tools/refine_native_panoptic.py \
  --mesh "$NEW_MESH_DIR/mesh.ply" --map "$NEW_MESH_DIR/map.ply" \
  --trajectory "$NEW_MESH_DIR/trajectory.json" --native-classes "$MAP_CLASSES" \
  --config "$ONEFORMER_SCAN20_CONFIG" --checkpoint "$ONEFORMER_SCAN20_CHECKPOINT" \
  --vote-views --save-view-evidence --output "$VOTED_OUTPUT"

python tools/extend_panoptic_surface.py \
  --mesh "$NEW_MESH_DIR/mesh.ply" --map "$NEW_MESH_DIR/map.ply" --anchor "$VOTED_OUTPUT" \
  --checkpoint "$PTV3_SCANNET_CHECKPOINT" --save-point-evidence --output "$SURFACE_OUTPUT"

python tools/refine_surface_boundaries.py \
  --map "$NEW_MESH_DIR/map.ply" --views "$VOTED_OUTPUT" \
  --surface "$SURFACE_OUTPUT" --output "$BOUNDARY_OUTPUT"
~~~

第三步只使用 CPU。它核对前两步的网格、点序及输出哈希，并写出同点序的 `map_labeled.ply`、`panoptic.npz` 与收据。可选 `--labels` 接受同一地图上其他已封存的 ScanNet20 实例标签。开发者也可以直接调用 `exchange_boundaries(xyz, semantic, instance, views, probabilities, nearest, distance)`。

在 9 场训练和 142 场验证的 63,336,701 个实际地图点上，发布的边界函数逐点复现了原实验的类别与实例。对 `scene0062_00` 重复两次实际四视角网络推理，开启和关闭视角保存产生相同的全景文件。接上实际 PTv3 和第三步后，94,618 点的类别、实例、坐标、RGB 与冻结结果逐点一致。[完整命令回归收据](scene0062_boundary_equivalence.json)

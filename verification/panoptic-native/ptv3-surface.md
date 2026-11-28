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

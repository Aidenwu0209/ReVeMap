# 原生地图上的 ScanNet20 全景精修

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

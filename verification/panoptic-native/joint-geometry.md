# 四视角实例投票与联合位姿图

在 142 条既定 ScanNet 官方验证扫描上，保持完整真值域、5 cm 投影和逐扫描等权。重融网格上同一组四视角投票使用双向 RGB-D 坐标时，PQ 为 45.83%。把同序实例投到联合位姿图修正的坐标后，PQ 为 46.37%，mIoU 为 56.21%，命中 1,065/2,116 个物体，误检 1,170 个。原 ReVeMap 主线为 24.78% PQ，逐场比较有 134 升、1 平、7 降。[逐扫描评分](remesh-vote142.csv) · [汇总与封存收据](remesh-vote142.json)

先运行重融命令，再运行带有 `--vote-views --native-classes` 的实例精修命令。下列两步读取产生这些实例的原始传感器记录与原估计轨迹，为投票结果生成新的点坐标。测量端需要能导入 MASt3R 作者代码的 PyTorch 环境。该端在 CPU 上运行。联合图和读写端沿用重融环境。[重融入口](README.md) · [四视角命令](ptv3-surface.md) · [MASt3R 作者代码](https://github.com/naver/mast3r)

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python tools/refine_scannet_loop_geometry.py measure \
  --sensor "$SCENE_SENS" --trajectory "$ESTIMATED_TRAJECTORY" \
  --checkpoint "$MAST3R_CHECKPOINT" --output "$LOOP_DIR"

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python tools/refine_scannet_loop_geometry.py apply \
  --sensor "$SCENE_SENS" --trajectory "$ESTIMATED_TRAJECTORY" \
  --map "$COMPLETED_MAP" --labeled "$VOTED_OUTPUT/map_labeled.ply" \
  --edges "$LOOP_DIR/edges.json" --output "$JOINT_OUTPUT"
```

`--map` 指向原 ReVeMap 完成补全后的 `A.ply`。分开传入同序投票 PLY，保留实例编号与 RGB，避免把半幅几何修正重复施加在同一地图上。评分时仍采用既定首对相机对齐。测量端从原始 RGB-D 解算跨帧运动，以独立像素区域的实测深度验证匹配，固定首帧后与短基线 RGB-D 因子联合求解。应用端校验原始传感器与轨迹哈希，并输出 `map_labeled.ply` 及 `result.json`。

本次使用 MASt3R-ViTLarge-BaseDecoder-512-catmlpdpt-metric 权重，SHA-256 为 `e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2`。开发扫描 scene0062_00 中，两个独立命令从传感器和原地图重现 523 条短基线边和 4 条长基线边，94,618 个点的坐标、类别、实例编号和 RGB 与封存实现逐点一致。[复现记录](packaged-joint-0062.json)

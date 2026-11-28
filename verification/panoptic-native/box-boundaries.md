# 用三维候选复核 RGB-D 实例边界

在同一个原生重融网格上运行四视角实例投票、PTv3 扩边和 ODIN 关联。运行 PTv3 时加上 `--save-point-evidence`，保留每个网格顶点的 20 类概率、地图点最近邻索引和距离。两个 ODIN 视角序列与三维 ISBNet 查询共同决定相邻实例边缘的点归属。[四视角与 PTv3 命令](ptv3-surface.md) · [ODIN 导出与关联命令](odin-surface.md)

三维查询导出需要安装 ISBNet 的推理代码、原生 CUDA 算子、`segmentator` 和 Open3D。作者提供的 ScanNet 训练配置位于 `configs/scannetv2/isbnet_scannetv2.yaml`，本次使用的权重 SHA-256 为 `fd2202d76ee7d8adda0d1f546a3bd7abce169871bcc41c8c0c1a3e1a75ba7576`。[ISBNet 作者实现及安装说明](https://github.com/VinAIResearch/ISBNet) · [作者发布的权重](https://huggingface.co/TuanNgo/ISBNet/blob/main/scannetv2/head_3dyco_scannetv2_val.pth)

```bash
python tools/export_isbnet_queries.py \
  --mesh "$NEW_MESH_DIR/mesh.ply" --map "$NEW_MESH_DIR/map.ply" \
  --anchor "$VOTED_OUTPUT" --config "$ISBNET_CONFIG" --checkpoint "$ISBNET_WEIGHTS" \
  --output "$OUT/box"

python tools/reconcile_panoptic_boundaries.py \
  --map "$NEW_MESH_DIR/map.ply" --anchor "$OUT/association/combined.npz" \
  --point-evidence "$SURFACE_OUTPUT/mesh.npz" --view19 "$OUT/queries/view19/queries.npz" \
  --cap64 "$OUT/queries/cap64/queries.npz" --box-queries "$OUT/box/queries.npz" \
  --output "$OUT/reconciled"
```

第二条命令在 CPU 上运行，在同一地图点序输出 `panoptic.npz`、带类别与实例的 PLY、类别表和输入输出 SHA-256。需要联合位姿修正时，把此 PLY 传给已有几何命令的 `--labeled` 参数。[联合几何命令](joint-geometry.md)

9 条官方训练扫描上，固定混合位姿和 5 cm 评测，增加三维证据前后的平均 PQ 为 61.01% 和 61.48%，命中由 85 增至 86，误检由 87 减至 85。4 条扫描提高，5 条下降。scene0140_00 提高 7.07 个百分点，对该组均值的影响最大。开发扫描 scene0062_00 与 scene0070_00 各从实测重融网格独立重跑两个命令，最终 94,618 与 338,213 个点的类别和实例编号分别逐点复现。scene0070_00 的中间查询槽位存在变化，最终标签 SHA-256 相同。[逐扫描开发结果](box-boundaries-training.csv) · [两次命令的实测收据](box-boundaries-replay.json)

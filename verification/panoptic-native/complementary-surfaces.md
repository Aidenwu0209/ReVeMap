# 用同次 RGB-D 补回主图漏掉的物体表面

在 SSH33 上，既定 142 条 ScanNet 验证扫描的平均 PQ 从原主线的 **24.78%** 提高到 **50.55%**。在已完成实例边界精修的主图上，仅加入另一条轨迹实际重建的表面，PQ 从 47.59% 提高到 50.55%，mIoU 从 57.67% 提高到 60.51%。[逐扫描配对及预测哈希](complementary-surfaces142.csv) · [评分与区间](complementary-surfaces142.json)

| 同一批扫描、同一评分器 | 平均 PQ（%） | 平均 mIoU（%） | 物体命中 / 2,116 | 物体误检 |
|---|---:|---:|---:|---:|
| ReVeMap 原主线 | 24.78 | 39.14 | 574 | 2,307 |
| 重融主图、四视角实例投票、PTv3 边界精修、联合位姿图 | 47.59 | 57.67 | 1,094 | 1,136 |
| 主图加辅助轨迹的实测表面 | **50.55** | **60.51** | **1,176** | **1,330** |

**哪些物体受益。** 90 个真值物体新达到匹配阈值，其中 88 个沿用了主图原有的实例编号。另有 8 个失去原匹配。新增匹配中有 43 个在主图上的真值表面覆盖率原本就不超过一半。在原图上单独改进这些物体的类别或实例边界，覆盖率会限制可达到的 IoU。相对精修主图，104 场 PQ 提高、4 场持平、34 场下降。相对原主线，138 场提高、1 场持平、3 场下降。[逐场、逐类匹配与误检分析](added-surfaces142.md)

**覆盖率与几何代价。** 新增 19,324,053 个实测点，完整标注真值顶点的 5 cm 覆盖率由 80.61% 提高到 86.82%。从地图向真值网格反向计算，落在真值顶点 5 cm 内的地图点比例由 52.75% 降至 47.84%，二者均为扫描等权均值。双向平均最近邻距离也从 14.69 cm 增至 15.67 cm。误检净增 194 个，其中“其他家具”净增 62 个、柜子 30 个、桌子 21 个和椅子 20 个。对需要高几何精度的使用场景，精修主图是可单独导出的结果。[三组共 426 个扫描结果的独立复算](complementary-surface-audit142.json) · [逐场几何](complementary-surface-audit142.csv) · [逐类误检](added-surfaces142-classes.csv)

## 复现流程

1. 从同一份 ReVeMap 估计轨迹、已补全的原生地图、彩色网格和原实例类别表出发。在 RGB-D 重融网格上运行 OneFormer3D 四视角投票，保留每个视角的证据。再运行 PTv3 表面扩展并保存逐点概率，最后运行 `tools/refine_surface_boundaries.py`。这三个命令分别生成 `$VOTED_OUTPUT`、`$SURFACE_OUTPUT`、`$BOUNDARY_OUTPUT`。[主图重融](README.md) · [含所有参数的三步命令](ptv3-surface.md#三维语义扩展和争议边界重分配)
2. 用原始 RGB-D 测量短基线运动与 MASt3R 闭环，运行联合位姿图的 `measure` 和 `apply`。在 `apply` 命令传入 `--map "$COMPLETED_MAP" --labeled "$BOUNDARY_OUTPUT/map_labeled.ply"`，得到 `$JOINT_OUTPUT/map_labeled.ply`。这一步保留实例编号和点序。[联合位姿图命令](joint-geometry.md)
3. 对同一条原始 RGB-D 序列运行 ORB-SLAM3。用两组估计轨迹的公共帧对齐，取 ORB 和原轨迹之间的半幅位姿，以 2 cm TSDF 重融，再在新表面上运行单次 OneFormer3D。使用 `--native-map "$COMPLETED_MAP"` 同时导出主图原有各点在辅助轨迹下的对应坐标。[辅助重融、模型和命令](orb-half-surface.md)
4. 运行下面的补点命令。它用两条轨迹下的对应点匹配同类实例编号。对已预测出有效类别、且原始测量位置离主图任一点超过 4.5 cm 的辅助点，保留其测量坐标与颜色并追加到主图末尾。补点距离固定为 4.5 cm，评分的投影距离始终为 5 cm。

```bash
python tools/fuse_complementary_surfaces.py \
  --primary "$JOINT_OUTPUT/map_labeled.ply" \
  --secondary "$ORB_OUTPUT/single/map_labeled.ply" \
  --primary-in-secondary "$ORB_OUTPUT/primary_in_secondary.ply" \
  --output "$FINAL_OUTPUT"
```

补点命令仅使用 NumPy、SciPy 和 plyfile，输出到新目录，包含 `map_labeled.ply`、`panoptic.npz`、`classes.json` 与输入输出哈希。SSH33 上从冻结的两组表面和对应坐标重新执行补点，142 场最终 79,043,367 个点的坐标、类别、实例编号与封存预测逐点一致。[逐扫描回放收据](complementary-replay142.json)

## 验证范围

要给导出的地图重新评分，按示例将 `--prediction` 指向 `$FINAL_OUTPUT/map_labeled.ply`、`--classes` 指向 `$FINAL_OUTPUT/classes.json`，轨迹使用原生估计轨迹。在 SSH33 上用发布的两条命令导出并评分 `scene0063_00` 的 592,615 个地图点，得到 PQ 56.9203%、mIoU 66.0441%、9 个命中和 9 个误检，与该场封存记录一致。[可复制的评分命令](full-domain-scoring.md) · [导出文件哈希](complementary-scene0063-export.json) · [单场独立评分](complementary-scene0063-score.json)

在 9 条与验证扫描房间家族互斥的官方训练扫描上比较了补点方案。保留辅助表面测量坐标的方案将该组 PQ 从 60.35% 提高到 62.14%，随后固定该方案与 4.5 cm 参数。[开发组逐扫描对照](complementary-development9.csv) · [验证前冻结的规则](complementary-rule.json)

验证使用官方 312 条验证扫描中项目既定的 142 条，涵盖 134 个房间家族。预测使用传感器数据和估计轨迹，全批预测封存后才读取评测真值。原主线、精修主图和补点输出对每场使用相同的首对相机对齐、完整标注真值顶点和 5 cm 最近邻投影。未覆盖的真值仍参与评分。按类别以严格大于 0.5 的 IoU 匹配实例，先在扫描内求 PQ，再对扫描等权平均。

同一验证子集在开发期间多次用于比较配置。按 134 个房间家族进行 10,000 次重采样，最终 PQ 均值的 95% 百分位区间为 46.99% 至 54.06%，补点相对精修主图的配对增量为 2.95 个百分点，区间为 1.87 至 4.08 个百分点。[逐扫描得分、封存哈希和重采样设置](complementary-surfaces142.json)

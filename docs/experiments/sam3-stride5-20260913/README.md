# SAM3 每 5 帧版本（2026-09-13）

按用户要求将本轮覆盖优先方案纳入 `developnew`。算法源码为
`ee16f96515ebd1fb328c3082bd44699833692bd5`，取帧采用 `stride=5`，保留首尾帧。
这是已验证的离线语义重放方案；`run-rgbd` 和原有 `run-semantic` 命令不会自动切换为 SAM3。

流程：冻结 RGB-D 轨迹与原地图 → SAM3 类别提示分割与 SGF 先验辅助的类别族修正
→ 深度一致投影与多帧语义投票 → 几何物体关联 → 多视图实例一致性
→ 带分裂掩码否决的实例恢复 → 原顶点上的语义/实例标签导出。
本轮物体关联使用几何证据，没有重新运行 SGA 神经网络；不采用 EVOL-SAM3，未运行 OneFormer。
源码中历史实验模块保留，但不代表全部启用。

## 选择依据和限制

| 场景 | 每 20 帧语义/实例覆盖 | 每 10 帧语义/实例覆盖 | 每 5 帧语义/实例覆盖 |
| --- | --- | --- | --- |
| Orbbec 4812 帧 | 35.8% / 16.9% | 46.5% / 34.3% | 54.3% / 46.6% |
| ScanNet0050 | 56.3% / 47.7% | 63.5% / 50.8% | 68.4% / 56.9% |
| ScanNet0011 | 50.2% / 39.1% | 59.9% / 45.8% | 69.1% / 52.9% |

这些是固定原地图上的正标签覆盖率，不是准确率或物体召回。
ScanNet0030 的独立取帧实验中，每 20/5/1 帧的类别正确实例 IoU≥0.5 匹配为
11/27、17/27、16/27；每 5 帧是本轮效果与成本的折中，不是所有场景的全局最优。
每 5 帧的累计选帧计算成本约为每 10 帧的两倍，不是完整 SLAM 端到端计时。

ScanNet0011 只有经过历史筛选的缓存参考标签：从每 20 帧到每 5 帧，条件语义一致率
89.47%→88.64%，实例纯度诊断 94.23%→93.72%。完整 GT 缺失，不能当作官方指标。
Orbbec 特殊设备仍存在漏标；3RScan 完整 RGB-D 不在当前可用路径，本轮没有完成其取帧对照。
本次合入表示采用覆盖优先配置，不表示所有准确率门槛通过。

## 运行入口与复现范围

[evidence/infer_frames.py](evidence/infer_frames.py) 和
[evidence/replay_arm.py](evidence/replay_arm.py) 是本轮实际执行脚本的逐字副本，
[evidence/sam3_multiview_blas.py](evidence/sam3_multiview_blas.py) 是等价整数计数加速实现。
它们依赖 ssh44 已有的模型、SGF 先验、原始输入和冻结源码目录，**不是独立下载即可运行的通用 CLI**。
模型/数据/地图不提交到 Git；依赖的实际路径与命令见 [运行记录](evidence/RUN_COMMANDS.md)
和各场景 `PLAN.json`。源码的 SAM3 模块随本次合入提交到仓库。

已有推理缓存的每 5 帧后端入口（使用新的场景输出目录）：

```sh
EXPERIMENT_SCENE_DIR=/absolute/path/to/new-scene-output \
  /home/aidenwu/Documents/sgf_sga_restore_20260910_v1/sga_env/bin/python \
  evidence/replay_arm.py --stride 5
```

该目录须具备该场景的 `PLAN.json`、`frames.jsonl`、`frames/*.npz` 和
`READY_stride5.json`。脚本校验缓存哈希，拒绝覆盖已有 `stride5` 目录。
重新推理时使用 `evidence/infer_frames.py`，它按 20/10/5 的嵌套选择生成共享缓存，
最终每 5 帧选取的预测与本轮一致；历史基线精确比较依赖 ssh44 已封存的基线文件。
不得把输出目录指向已封存的实验目录。

输出为 `stride5/map_labels.npz`、`objects.json`、`classes.json` 和恢复审计。
原几何上的带标签 PLY 已在下列完整实验目录导出：

- 本机：`/Users/wu/Desktop/wu/SGF-SGA/comparisons/cross_dataset_frame_ablation_20260913_v1`
- ssh44：`/mnt/d/SGF-SGA-experiments/cross_dataset_frame_ablation_20260913_v1`
- 每场景主地图：`<dataset>/<scene>/stride5/map/map_labeled.ply`

## 证据保存

[PROMOTION.json](PROMOTION.json) 记录此次用户指定合入、选择的配置和打包文件哈希。
`evidence/` 保存原始计划、9 组结果、输入/输出校验与完整封存文件清单。
原始记录里的 `adopted:false`、`default_promoted:false` 和 `quality_accepted:false`
反映实验封存时状态，原样保留；本次合入决策单独记录，不改写历史结果。
`scannet0030/` 保存三个取帧组的语义与类别正确实例评估。
不重新执行 GPU 推理；本次提交重新检查代码测试、原封存文件哈希和打包副本。

# ReVeMap 性能优化发布与正式服务回放验收 · 2026-09-24

已完成代码推送、主分支合并、两路正式扫描服务更新，以及最新 961 帧录制的完整重处理和结果读回。

## 已发布版本

- PR：[ReVeMap #9](https://github.com/Aidenwu0209/ReVeMap/pull/9)，已合并。
- 合并提交：`3683f6b902a1891da5a85d438f37b5461910a63f`。
- 发布代码提交：`0a7a4457b712abdd0936fa17d7d3fb5582a710d7`；其 Git tree 与合并提交完全一致。
- iPad 服务 8765 和无线服务 8766 均运行 `/home/aidenwu/Documents/ReVeMap-unified-20260923/releases/0a7a445/src`。
- 桌面 `启动统一ReVeMap.command --check` 已确认连接到这两个新进程。原始录制和旧 attempt 保留。
- API 的 `base_commit` 仍是历史常量，不作为当前发布版本证据；以实际进程路径、发布文件摘要和本轮 CHECKPOINT_CONTEXT 为准。

## 同一录制的处理耗时

录制 `scan_20260924_170000_b22854`：961 帧，stride=5，193 个语义视图；本地 Qwen3-VL-2B-NF4，串行调度，开启 refinement。新运行通过正式服务的 `/api/reprocess` 创建独立 attempt，mapping 和 SAM3 均未复用旧结果。

| 范围 | 旧发布版 | 新发布版 | 变化 |
|---|---:|---:|---:|
| 完整处理，含 refinement 与导出 | 739.54 s | 709.41 s | -4.07% |
| 主流程，不含后续 refinement 等 | 667.15 s | 638.62 s | -4.28% |
| 轨迹与几何 | 176.77 s | 175.39 s | -0.78% |
| SAM3 | 369.27 s | 343.70 s | -6.92% |
| 主 VLM | 94.90 s | 94.22 s | -0.71% |
| 语义融合 | 17.61 s | 17.06 s | -3.14% |

各阶段包含于总时间内，不能将全表相加。完整处理减少 30.13 秒（4.07%）。这是同一录制在不同时刻的一次重跑比较，文件缓存、硬件状态和测量开销可能影响数值，不能将所有时间差归因于本次优化，也不是多次实验的稳定速度结论。本轮未测进程最终峰值 RSS；此前约 41% 的 RSS 降幅来自单独 SAM3 对照。

## 正确性与生效证据

- 已重新校验 5733 个原始采集和旧 attempt 文件，摘要全部未变。
- 新运行记录的源码摘要与发布包匹配，包含新增 sam3_text_cache.py。
- `checkpoint_mmap=true`、`checkpoint_assign=true`；缺失权重列表为空。
- 文本缓存命中 5952 次、首次编码 31 次，缓存 4572128 字节。
- 193 个语义视图、965 个数组逐元素和 dtype 一致；1250 张裁图逐字节一致。
- 1250 条主 VLM 裁图结果的图片摘要、名称、格式状态与原始回答一致。
- 轨迹覆盖 961/961 帧，全部有效；最终点云 410605 点且坐标有限。
- 最终 ARTIFACTS 清单的所有引用文件摘要通过核验。
- `mapping/refill/final_raw_poses.npy`：一致。
- `fused/map_labels.npz`：一致。
- `fused/export/map_labeled.ply`：一致。
- `refinement/refined/capture/map_labels.npz`：一致。
- `refinement/refined/capture/semantic_labeled.ply`：一致。

两路正式服务均读回本次新 attempt；场景图包含 69 个对象、482 条关系。计数查询与场景图对象数一致，查询和图导出均绑定最终地图 SHA256；表面接口也已检查。这些检查验证部署和结果一致性，不等于新的人工真值准确率评测，也不替代一次新的物理相机采集或 iPad 手势验收。

## 测试与回退

- 本地 Python：429 passed、3 skipped、24 subtests passed；跳过项包括本地没有 PyTorch/Open3D 的测试。
- 计算主机真实 PyTorch 环境：缓存、加载器、runtime、checkpoint 共 48 passed。
- 浏览器原生交互桥：5 passed。
- PR 的 Python 3.11/3.12 CI 及合并后的主分支 CI 均成功。
- 正式配置保留每个服务的旧版 rollback 入口。若需回退，使用原发布器的 `ipad rollback` 和 `wireless rollback`，发布器会拒绝打断活跃扫描。备份配置为 `/home/aidenwu/Documents/ReVeMap-unified-20260923/services.before-performance-0a7a445.json`。

机器可读摘要见 [发布验收回执](performance-deployment-20260924.json)。原始发布、运行和读回回执保存在本地 `性能优化发布_20260924/` 及计算主机 `/home/aidenwu/Documents/ReVeMap-unified-20260923/`。

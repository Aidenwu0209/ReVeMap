# ReVeMap 性能优化与同输入验证 · 2026-09-24

本轮完成 SAM3 固定文本编码复用和模型加载内存优化。在 RTX 4060 Laptop 8 GB 上，用同一段真实 352 帧输入、原有 stride=5 的全部 72 个语义帧进行新旧对照：SAM3 阶段耗时减少 **6.2%**，该进程峰值常驻内存减少 **41.0%**。这是单场景、单次完整配对的阶段结果，不是整个系统端到端提速比例。

代码基于 `main@a468bb2`，位于独立分支 `codex/pipeline-performance-20260924`。这份报告记录部署前的独立验证；后续已完成正式服务更新，见[发布与完整回放验收](performance-deployment-20260924.md)。

## 测量结果

| 指标 | 原版 | 优化版 | 解释 |
|---|---:|---:|---|
| SAM3 阶段墙钟 | 142.73 s | 133.86 s | 包含模型加载、推理和输出；不含 Python/torch 初始导入 |
| 加载后处理 | 134.83 s | 127.08 s | 相同 72 帧、31 个有效文本类别 |
| CPU 时间 | 144.69 s | 135.66 s | 进程 user+system，包含启动导入，不能与墙钟直接作利用率比值 |
| 进程峰值 RSS | 7.28 GiB | 4.30 GiB | 内存下降；不是显存指标 |
| CUDA 峰值 allocated | 5088.42 MiB | 5092.79 MiB | 缓存使显存增加约 4.37 MiB |
| CUDA 峰值 reserved | 5332 MiB | 5336 MiB | 分配器保留空间 |
| 文本编码次数 | 2232 | 31 | 2201 次命中；缓存数据 4,572,128 字节 |

计时比较关闭 cProfile，两个版本顺序运行，使用相同模型权重、精度、随机种子、线程设置、图片和深度。没有降低选帧数量、图像分辨率或类别数量。显存以 PyTorch allocator 指标计量，不包括 CUDA context 等全部驱动占用。9 帧带 profiler 的初测仅用于定位，不用于最终提速百分比。

[结构化结果](performance-20260924.json) 包含源文件哈希、输入清单哈希、峰值指标与匹配回执。完整本地诊断资料位于 `profile_output/remote/`；远端原始数组、裁图和日志位于 `/home/aidenwu/Documents/ReVeMap-performance-20260924/profile_output/`。

## 修改内容

1. **固定文本特征复用。** 在单个 SAM3 worker 内给模型实例的 `forward_text` 加有界缓存，最多 64 条、16 MiB。只处理无几何提示、无附加文本的固定模型推理调用；按文本、设备、autocast 状态和精度区分。返回张量副本，防止后续操作修改缓存；异常或正常退出均恢复原方法并释放缓存。Provider 的分割与 grounding 路径保持原样。
2. **减少权重的 CPU 重复存储。** 先照常校验模型 SHA256，再对现代 checkpoint 使用 `mmap`。类型与布局一致时用 `assign=True` 替换初始化存储，随后转入 CUDA；需要类型转换或旧式 checkpoint 时沿用拷贝语义。缺失权重继续拒绝执行。此改动也作用于使用同一个加载器的 refinement；本轮没有单独量化 refinement 性能。

没有改动轨迹、TSDF、语义判定阈值、实例融合或场景查询。保留每帧 CUDA 空闲缓存清理与模型进程隔离；本轮主要收益来自消除重复文本编码和权重存储。

## 验证与范围

- 72 帧全部 360 个数组逐元素、逐 dtype 一致，覆盖原始压缩 mask、语义、实例、置信度和深度尺寸。
- 576 张裁图文件逐字节一致；裁图任务、区域、对应类别和原始 RGB/深度摘要一致。
- 反向顺序复测（先优化版、后原版；9 帧；无 profiler）：23.56 s 对 25.19 s，分别约 4.30 GiB 对 7.28 GiB；45 个数组和 72 张裁图一致。该短序列仅用来检查运行顺序影响。
- 保持原始几何与轨迹，使用新生成的 72 帧语义证据重新融合，semantic/instance/confidence 完全一致，补全前融合点云 SHA256 均为 `0ef4281a3793d33ed26c99eb9e55fdefe75428a3370aed103e86444ff71a78d4`。没有重新运行 SLAM、VLM 或补全后地图。
- 新缓存与加载器的 11 项测试在真实 PyTorch 2.7.1 环境通过，覆盖缓存污染隔离、容量限制、精度区分、旁路、异常恢复、旧格式和类型转换，以及缺失权重拒绝。
- 本地既有回归 429 passed、3 skipped、24 subtests passed；3 个 skip 包含无 Open3D 的 2 项和无 PyTorch 的新增测试模块，新增模块已在远端补测。加载器最终调整后的针对性回归 30 passed。
- Wheel 构建成功；在 checkout 外从独立安装目录导入新模块，并核对其源文件与本次代码一致。

完整 pipeline 的 SLAM、VLM、可选 refinement 与发布阶段未重新做端到端速度测量，不能把 6.2% 直接称为全流程收益。未测 GPU kernel 占用、PCIe 带宽和整机峰值内存。

## 瓶颈与后续顺序

本次读回的历史串行日志中，352 帧主流程约 299.62 s，SAM3 161.94 s、建图 68.80 s、VLM 52.11 s；1005 帧主流程约 742.41 s，SAM3 433.02 s、建图 170.24 s、VLM 110.02 s。这些旧日志只用于定位阶段，未用于计算本轮收益，且主流程时间不含后续 refinement。重复的 VLM complete 事件只采用第一次。

下一轮最值得测现有 `parallel` 调度对 CPU TSDF/SAM3、CPU 融合/VLM 的重叠收益；然后分别剖析 VLM 命名与可选补全。进一步减少帧或改模型应另立质量对照，不应混进这次等输出优化。

## 测量代码与改动清单

| 文件 | 类型 | 用途 |
|---|---|---|
| `tools/profile_sam3.py` | 新增测量脚本 | 独立运行 SAM3；可选 cProfile；墙钟、CPU、RSS、CUDA 峰值和源文件摘要 |
| `tools/compare_sam3_outputs.py` | 新增校验脚本 | 数组、输入摘要、裁图任务和裁图文件一致性检查 |
| `tools/replay_profile_fusion.py` | 新增校验脚本 | 固定旧几何，用新语义证据融合并比较地图 |
| `profile_output/` | 新增且 Git 忽略 | 剖析日志、pstats、各轮测量回执、本地源码回执、构建包；可单独清理，不影响生产代码 |
| `src/pose_pipeline/sam3_text_cache.py` | 新增生产代码 | 有界文本特征缓存 |
| `src/pose_pipeline/sam3_mapping.py` | 修改生产代码 | mmap/assign 权重加载及 MODEL 审计字段 |
| `src/pose_pipeline/semantic_runtime/worker.py` | 修改生产代码 | 缓存作用域及 COMPLETE 中缓存计数 |
| `tests/test_sam3_text_cache.py` | 新增测试 | 11 个 CPU 张量与 checkpoint 用例，需要 PyTorch |
| `.gitignore` | 修改 | 忽略测量产物 |
| `README.md` | 修改 | 链接性能结果及其适用范围 |
| `docs/performance-20260924.md`、`.json` | 新增 | 报告及紧凑回执 |

测量代码全部在独立脚本；生产逻辑没有插入 cProfile 或资源采样。需要时可单独清理这些测量脚本与 `profile_output/`。

## 复现

在计算主机的 SAM3 环境执行，分别将 `PYTHONPATH` 指向原版和优化版源码，每轮输出用新目录：

```bash
PYTHONPATH=/absolute/path/to/src /absolute/path/to/sam3-python \
  tools/profile_sam3.py \
  --manifest /absolute/path/to/manifest.json \
  --runtime /absolute/path/to/runtime.json \
  --output /absolute/path/to/new-output --stride 5

/absolute/path/to/sam3-python tools/compare_sam3_outputs.py \
  /absolute/path/to/baseline-output /absolute/path/to/candidate-output \
  --output /absolute/path/to/new-parity.json
```

公平计时关闭 `--profile`，保持相同线程数，避免其他 GPU 工作并发。`--profile` 仅用于热点诊断。

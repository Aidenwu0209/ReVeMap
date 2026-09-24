# 多场景验证与 VLM 瓶颈剖析 · 2026-09-24

> 本报告记录部署前的多场景阶段对照；后续正式服务更新及 961 帧完整回放见[发布验收](performance-deployment-20260924.md)。

新增 **3 个 ScanNet 场景窗口和 1 段完整 1005 帧 Orbbec 录制**，原版与优化版的 SAM3 输出及固定几何上的融合地图均完全一致。SAM3 阶段耗时下降 **6.28%–6.89%**，进程峰值 RSS 下降约 **41%**。VLM 确实占用可观时间：两段近期录制中占主流程约 **15%–17%**；但 SAM3 的时间占比更高。

离线 VLM 批处理试验得到提速，同时出现名称变化，因此本次没有将批处理接入生产。输出变化尚不能判定为准确率升降，但已不满足这轮等输出优化的条件。

代码对照为 `a468bb2` → `aaf0c2f`，运行源码 SHA256 与这两个提交逐一匹配。本次只增加验证和剖析脚本；原有生产服务、模型、精度、分辨率和阈值没有调整。完整数值、路径和摘要见 [结构化回执](performance-multiscene-20260924.json)。

## 新增同输入对照

| 输入 | 本轮 RGB-D 范围 | SAM3 视图数 | 原版 | 优化版 | 耗时下降 | 语义与融合输出 |
|---|---|---:|---:|---:|---:|---|
| ScanNet scene0011_00 | 帧 1127–1246，共 120 帧 | 25 | 56.05 s | 52.48 s | 6.37% | 完全一致 |
| ScanNet scene0030_00 | 帧 1189–1308，共 120 帧 | 25 | 58.73 s | 55.04 s | 6.28% | 完全一致 |
| ScanNet scene0050_00 | 帧 2266–2385，共 120 帧 | 25 | 61.81 s | 57.55 s | 6.89% | 完全一致 |
| Orbbec 1005 帧录制 | 帧 0–1004，完整录制 | 202 | 385.84 s | 360.66 s | 6.53% | 完全一致 |

三个 ScanNet 窗口预先取各自序列中部，并非整条 ScanNet 序列。原始序列分别为 2374、2498、4652 帧。Orbbec 的另一段 352 帧对照是[上一轮结果](performance-20260924.md)，本轮未重复计时；两段 Orbbec 录制不算两个已确认独立房间。

统一使用 RTX 4060 Laptop 8 GB，stride=5，线程数为 2，关闭 cProfile。每组包含模型加载、推理、保存，不含初始 Python/torch 导入。每组只做一次完整配对，运行顺序按 A/B、B/A、A/B、B/A 交替，GPU 工作串行执行；这些比例不代表统计置信区间或完整 pipeline 的提速。

| 内存指标 | 原版范围 | 优化版范围 | 含义 |
|---|---:|---:|---|
| 进程峰值 RSS | 7452.88–7454.13 MiB | 4397.46–4398.95 MiB | CPU 常驻内存约 7.28 → 4.30 GiB |
| CUDA 峰值 allocated | 5088.42 MiB | 5092.79 MiB | 增加 4.37 MiB，来自文本缓存 |
| CUDA 峰值 reserved | 5332–5348 MiB | 5336–5352 MiB | PyTorch 分配器保留量 |

这些是单进程和 PyTorch allocator 指标，不是整机内存或驱动级显存峰值。本轮没有测 SM 占用或 PCIe 带宽。

## 输出与输入保护

- 新增 277 个 SAM3 视图，1385 个数组逐 dtype、逐元素一致；1591 张裁图逐字节一致，裁图任务、类别、边界框及 RGB/深度摘要也一致。
- 四组 semantic、instance、confidence 融合数组一致，四对导出 PLY 的 SHA256 一致。使用保存的轨迹和几何，两个版本分别读取本轮新生成的语义证据进行融合。
- 原始 manifest、轨迹、点云及本轮 RGB/深度文件前后 SHA256 一致。导出回执明确标为固定几何上的部分流程重放，未冒充新跑完的完整 pipeline。
- 补测 352 帧录制的 refinement：固定对象和 VLM 命名决策，分别重新执行 grounding 和 apply，530 个数组以及最终 `semantic_labeled.ply` 完全一致。其 SHA256 为 `87b6dfbf222cbc9aad3f1e20d66de6635263c689dd9af744a7d1f004ad300043`。原 refinement 输入未改变。
- 完成额外的双向文件集合、数组键及内容检查，覆盖四组融合目录和 refinement 的 grounding/refined 目录，均通过。
- 本轮针对语义运行、融合与 refinement 的 35 项已有回归通过；四个新脚本语法检查通过。此前缓存和加载器的真实 PyTorch 测试见上一轮报告。

这些结果支持“本次工程优化在已测输入上保持输出”，并不提供新的 GT 精度、实例 AP 或语义 IoU。三个 ScanNet 窗口只覆盖各自部分视角，不能据此评价整场景语义完整度。本轮未重新运行 SLAM、完整场景图/查询流程或全流程计时。Refinement 检查固定了原 VLM 决策，不能替代新 VLM 策略的端到端质量验证。

## VLM 是否是速度问题

是，当前逐张命名会累积明显耗时。读取的 22 份历史回执用于定位，下面列出两段近期录制；它们不是本轮新的端到端计时。

| 历史输入 | 主流程耗时 | 主 VLM worker | 主流程占比 | 裁图次数 | 单张平均 |
|---|---:|---:|---:|---:|---:|
| 352 帧 | 299.62 s | 51.13 s | 17.06% | 576 | 78.8 ms |
| 1005 帧 | 742.41 s | 109.04 s | 14.69% | 1278 | 80.8 ms |

主流程总时间不含后续 refinement；refinement 另有 146/278 次命名，分别用时 10.48/20.78 s（不含加载）。历史 SAM3 子进程分别为 161.94/433.02 s，占主流程约 54%/58%。即使将整个主 VLM 阶段减半，在其他阶段不变的串行假设下，主流程耗时也只会下降约 7%–9%；这只是按占比计算的情景，不是已测收益。

本轮重新运行 352 帧的全部 576 次 VLM 请求，模型为 `Qwen/Qwen3-VL-2B-Instruct`、NF4，权重 revision 为 `89644892e4d85e24eaac8bacfd4f463576704203`。576 条名称、原始响应、valid 标志和图片摘要与原记录全部一致。

| 剖析项目 | 次数 | 耗时 | 计时关系 |
|---|---:|---:|---|
| 完整 worker | 1 | 65.60 s | 含 cProfile 与同步开销 |
| 创建命名器 | 1 | 16.30 s | 包含导入、权重校验、模型构建/加载 |
| 权重完整性校验 | 1 | 9.27 s | 已包含在上一行，不能相加 |
| 模型加载审计值 | 1 | 2.29 s | 仅加载部分，不能代表全部启动成本 |
| 请求合计 | 576 | 49.04 s | 包含下列三个部分及其他请求操作 |
| 图文预处理 | 576 | 1.03 s | 请求内部 |
| generate | 576 | 47.23 s | 包含视觉编码、prefill、解码和等待 |
| 文本解码 | 576 | 0.026 s | 请求内部 |

剖析运行峰值 RSS 为 5137.93 MiB，CUDA allocated 为 1576.36 MiB。cProfile 是 Python 调用剖析，`generate` 墙钟不能全部解释成 GPU 计算时间。此次启动读取权重耗时明显，受文件缓存与 I/O 状态影响；不可拿 65.60 s 与历史 51.13 s 直接宣称性能退化。

两个近期主 VLM 输入中，完全相同图片 SHA 的额外请求仅为 0/2 次，精确图片缓存的直接收益有限。平均生成约 2.3 tokens，最长 4 tokens，当前上限为 24；缩短上限不是现有数据支持的主要提速方向。按对象合并“看起来相似”的裁图会改变证据，需要独立质量验证。

## 离线批处理试验

从 576 张裁图中均匀选取 96 张，加载模型一次并预热，按 batch=1、2、4、1 顺序执行。提示词、图片、权重、精度和 greedy 生成参数相同；批处理使用左侧 padding。记录的是热模型下 96 张裁图的处理时间，不含启动，也不是整个 pipeline。

| 批大小 | 时间 | 相对第一次逐张处理 | 名称变化 | CUDA allocated 峰值 |
|---|---:|---:|---:|---:|
| 1 | 6.86 s | 基线 | 0/96 | 1576.36 MiB |
| 2 | 5.04 s | 耗时下降 26.5% | 1/96 | 1615.03 MiB |
| 4 | 4.48 s | 耗时下降 34.7% | 3/96 | 1684.05 MiB |
| 1，再次执行 | 6.72 s | 顺序/重复检查 | 0/96 | 1579.20 MiB |

batch=2 将 `green curtain` 改为 `unknown`；batch=4 还出现 `cabinet` → `shelves`、`unknown` → `cabinet`。两次逐张结果完全一致，所有输入摘要不变。尚未定位差异来自 padding、批内形状还是数值路径，也没有 GT 证明哪种名称更准确。因此本次保留现有逐张命名，批处理脚本仅用于离线实验。

## 后续优化顺序

1. **先测现有阶段调度的端到端收益。** SAM3 占比仍最大，CPU 建图/融合与 GPU 阶段是否能有效重叠，需要完整计时与输出一致性对照。
2. **检查 VLM worker 生命周期。** 同一任务主命名与 refinement 重复创建模型，可能复用已完成完整性校验的实例；但需实测内存占用与模型切换，不能直接去掉权重校验。
3. **修正并扩大批处理验证。** 先查输入张量、padding 和数值差异，再对更多场景完整裁图及最终命名/地图验证。当前样本提速不能直接推广到完整运行，也不能据此启用默认批处理。

## 关于改用 DeepSeek 或 GPT API

本轮实测的 VLM 是本地 Qwen3-VL-2B-NF4，API key 不参与该模型推理。云端替换需要同时选择服务地址、图像模型与相容的请求参数。

现有 registry 已有 `deepseek_v41_flash_api`，指向 `deepseek-flash`，通过环境变量读取凭据。[DeepSeek 当前官方视觉文档](https://api-docs.deepseek.com/guides/vision/)明确支持图片输入；这是服务别名，不是固定模型版本。[OpenAI 官方视觉文档](https://developers.openai.com/api/docs/guides/images-vision)同样支持向视觉模型发送图片并返回文本。当前 API adapter 带有服务商专用的 `thinking` 等字段，GPT 需要适配请求，不能只替换 key 或地址。

云端模型可能改善某些物体名称，但本轮没有运行任何新的云端推理或进行人工标注评估，不能断言哪家更准或更快。本地当前单张平均约 80 ms；云端串行调用增加图片上传、网络往返和服务端等待，可能更慢。独立请求可以尝试有界并发，思路见 [OpenAI 延迟优化文档](https://developers.openai.com/api/docs/guides/latency-optimization)，实际吞吐还受账户限额和失败重试影响。

建议以本轮裁图作为固定输入，对本地、DeepSeek、GPT 分别记录完整阶段墙钟、p50/p95、失败/截断、费用与人工核验的类别正确率，再检查最终对象名称和查询结果。全量替换和“本地命名、疑难对象云端复核”是两种待验证策略；后者可能限制请求数，但要验证疑难样本筛选是否漏掉自信的错误。换命名模型不能直接修复缺失深度、几何重影或错误 mask。

## 产物与测量改动清单

本地摘要在 `profile_output/remote/`；远端原始数组、裁图、日志和剖析文件在 `/home/aidenwu/Documents/ReVeMap-performance-20260924/profile_output/`。结构化回执记录关键文件及四个测量脚本的 SHA256。

| 文件 | 变更类型 | 内容 | 行范围 |
|---|---|---|---|
| `tools/validate_semantic_scenes.py` | 新增 | 同输入 SAM3/固定几何融合对照、输入保护及范围声明 | 全文件 |
| `tools/validate_refinement_replay.py` | 新增 | 固定命名决策的 grounding/apply 对照与原文件保护 | 全文件 |
| `tools/profile_vlm.py` | 新增 | 独立 wrapper、cProfile、分段计时、原响应比较 | 全文件 |
| `tools/probe_vlm_batches.py` | 新增 | 96 张裁图离线批处理及重复逐张对照 | 全文件 |
| `profile_output/run_followup.py` | 新增、忽略 | 等待多场景完成后串行运行三项后续验证 | 全文件 |
| `profile_output/vlm-audit-summary.json`、`report-template-notes.md` | 新增、忽略 | 历史回执摘要及报告范围笔记 | 全文件 |
| `profile_output/remote/multiscene/**` | 新增、忽略 | 各组输入摘要、计划、测量、输出对照与完成回执 | 生成文件 |
| `profile_output/remote/refinement-pair/**` | 新增、忽略 | refinement 对照结果 | 生成文件 |
| `profile_output/remote/vlm-profile/**`、`vlm-batches/**` | 新增、忽略 | VLM 响应、模型回执、剖析及批处理结果 | 生成文件 |
| `profile_output/remote/*AUDIT.json`、`*COMPLETE.json`、`vlm-*.json` | 新增、忽略 | 双向文件/键审核、任务状态及历史日志审计 | 生成文件 |
| `docs/performance-multiscene-20260924.md`、`.json` | 新增 | 报告和结构化回执 | 全文件 |
| `README.md` | 修改 | 链接新增验证和 VLM 结果 | 性能验证段 |

测量只在独立脚本中包装函数；本轮没有给生产代码插入 profiler。完成审阅后，可按需清理这些测量脚本和 `profile_output/`，生产优化与原始扫描不会受影响。

## 复现入口

在远端实验根目录使用保存的 `profile_output/multiscene/PLAN.json`。每轮必须使用新的输出目录，保持相同运行环境和源代码快照：

```bash
PYTHONPATH=/absolute/path/to/candidate-src /absolute/path/to/sam3-python \
  tools/validate_semantic_scenes.py --plan /absolute/path/to/PLAN.json \
  --output /absolute/path/to/new-multiscene

PYTHONPATH=/absolute/path/to/candidate-src /absolute/path/to/sam3-python \
  tools/validate_refinement_replay.py --plan /absolute/path/to/PLAN.json \
  --reference /absolute/path/to/original/refinement \
  --output /absolute/path/to/new-refinement-pair

PYTHONPATH=/absolute/path/to/candidate-src /absolute/path/to/vlm-python \
  tools/profile_vlm.py --tasks /absolute/path/to/CROP_TASKS.json \
  --runtime /absolute/path/to/runtime.json --reference /absolute/path/to/original/vlm \
  --output /absolute/path/to/new-vlm-profile

PYTHONPATH=/absolute/path/to/candidate-src /absolute/path/to/vlm-python \
  tools/probe_vlm_batches.py --tasks /absolute/path/to/CROP_TASKS.json \
  --runtime /absolute/path/to/runtime.json --count 96 \
  --output /absolute/path/to/new-vlm-batches
```

多场景脚本依赖同目录下上一轮的 `profile_sam3.py` 与 `compare_sam3_outputs.py`。SAM3 与 VLM 使用各自已验证的 Python 环境；不要并发运行 GPU 试验。远端此次复制的脚本直接放在实验根目录，命令中的 `tools/` 前缀需相应去掉。

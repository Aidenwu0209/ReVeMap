# ReVeMap

RGB-D 多视角语义建图、实例恢复与可解释对象查询。原始 RGB-D 用于轨迹估计和几何重建，SAM3 观察经过深度检查、多视角融合及实例关联后写入地图。VLM 名称默认作为观察元数据；可选 refinement 和离线 P2 有独立入口。

## 性能验证

SAM3 worker 复用固定文本特征，并以 mmap/assign 降低权重加载的 CPU 内存。[首轮 352 帧对照](docs/performance-20260924.md)后，新增 3 个 ScanNet 的 120 帧窗口和一段完整 1005 帧录制：SAM3 阶段耗时减少 6.28%–6.89%、进程峰值 RSS 降低约 41%，掩码、裁图和固定几何上的融合地图一致。VLM 批处理离线试验虽有提速，但出现名称变化，未启用。验证范围和瓶颈分析见 [多场景与 VLM 报告](docs/performance-multiscene-20260924.md)。

[云端 VLM 对照](docs/cloud-vlm-20260924.md)覆盖同一批 24 张裁图：DeepSeek Flash 非思考、4 并发为 6.60 s，本地 Qwen 热模型为 2.26 s（含初始化 7.10 s）；用户提供的 GPT-6 Sol 中转六组存在 26/144 次传输失败，且请求 `none` 回传为 `medium`。这轮只测试命名接口，未切换生产模型，也未建立 GT 精度结论。

## CPU 安装与最小示例

需要 Python 3.11 或 3.12。以下示例不需要相机、GPU、模型权重或数据集。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
revemap --help
revemap demo --output outputs/synthetic-demo
revemap query-scene --graph outputs/synthetic-demo/scene_graph.json --label 椅子
revemap query-scene --graph outputs/synthetic-demo/scene_graph.json --label cup --relation supported_by --reference-id 1
```

示例生成三个人工几何对象：桌子 `1`、杯子 `2`、椅子 `3`。第二个查询应返回实例 `2`。`result.json` 明确标注 synthetic，`ARTIFACTS.json` 标注未绑定真实 RGB-D 来源。**示例只验证安装、产物读取及查询功能，不是模型推理结果或精度证据。** 输出目录必须不存在，避免覆盖已有结果。

## 原始 RGB-D 建图

复制 `configs/semantic_runtime.example.json` 为被 Git 忽略的 `configs/semantic_runtime.local.json`，填写本机的 CPU/GPU/SAM3/VLM 解释器、DROID-W 源码、模型权重路径。外部 provider、权重和相机 SDK 不包含在本仓库，也不会由普通安装自动下载。CPU 几何运行环境另需安装 `.[geometry]`；其他模型环境按对应模型/provider 的安装要求独立准备。

输入格式为 `rgbd_sequence_manifest.v1`，轨迹使用米制 `T_world_camera_m`。相机采集会生成清单；已有数据需通过 `pose_pipeline.contracts` 构建和验证，不能把数据集 GT 轨迹传入推理路径。

```bash
revemap run-sam3 \
  --manifest /absolute/path/to/manifest.json \
  --runtime configs/semantic_runtime.local.json \
  --output outputs/run-stride \
  --schedule serial --vlm none --stride 5
```

默认保留原有固定步长选帧及融合判定。三个冻结输入场景中，新默认与原版导出的 PLY 字节一致；这项验证说明兼容性，不代表新的精度提升。

下面的实验模式在最终回填轨迹可用后，用图像清晰度、有效深度比例和位姿差异选取语义帧；**几何重建仍使用全部原始帧**。

```bash
revemap run-sam3 \
  --manifest /absolute/path/to/manifest.json \
  --runtime configs/semantic_runtime.local.json \
  --output outputs/run-diverse \
  --schedule serial --vlm none \
  --view-policy quality-diverse --view-budget 60
```

预算不得超过输入帧数。公平比较时，`stride`、`quality` 和 `quality-diverse` 使用相同的显式 `--view-budget`。视角多样性不足时会按质量补足预算，降级帧记录在 `view_selection/VIEW_PLAN.json` 中；位姿差异不等于统计独立，本轮三个场景的同预算对照均未支持默认启用：`quality` 与 `quality-diverse` 的平均 unknown-as-error 准确率分别下降约 0.945 和 0.471 个百分点。

置信度和冲突策略也分开选择：默认 `--semantic-confidence-policy legacy --semantic-conflict-policy consensus` 保留原判定。`track` 修复跨类别共享最高分的分数归属问题，但单独重放仍出现平均约 0.027 个百分点的准确率下降；`abstain` 会额外撤回冲突标签，可能包括正确标签。两者目前只适合显式对照实验。保留 legacy 意味着其已知的共享分数局限仍然存在。

### 已完成阶段复用

```bash
revemap run-sam3 --manifest /absolute/path/to/manifest.json \
  --runtime configs/semantic_runtime.local.json --output outputs/attempt-1 \
  --schedule parallel --vlm none --checkpoint-stages
# 中断后，保持输入、代码、环境和算法选项相同，输出必须使用新目录。
revemap run-sam3 --manifest /absolute/path/to/manifest.json \
  --runtime configs/semantic_runtime.local.json --output outputs/attempt-2 \
  --schedule parallel --vlm none --resume-from outputs/attempt-1
```

只复用已封存的完整建图和 SAM3 阶段，校验原始帧、选项、源码、外部模型代码/权重、解释器包版本及输出 SHA256。部分阶段、旧版无检查点、文件变化或选项变化会重新计算；原尝试不会被修改。VLM、融合、命名和可选 refinement 会重新运行。`RESUME.json` 记录实际复用情况，复用运行的 `raw_fps` 为 null，避免与完整建图吞吐混用。

## 结果与评价

`ARTIFACTS.json` 统一记录地图、类别表、结果、输入清单、轨迹等文件的位置和 SHA256。原始流程命名完成后的清单位于 `fused/ARTIFACTS.json`；GUI 的最终清单位于当前处理尝试的 `pipeline/ARTIFACTS.json`。消费者应读取清单，不要手动移动 PLY 或拼接不同运行的文件。

```bash
revemap evaluate-semantic \
  --result outputs/run-stride \
  --reference-dir /absolute/path/to/scannet-or-3rscan-reference \
  --output outputs/evaluation-stride
```

输入清单中的序列目录应保留数据集评价所需 GT pose；这些文件只在已完成预测后的评价阶段读取。旧结果未绑定输入来源时，必须显式使用 `--allow-unbound-legacy` 并提供 `--manifest`、`--trajectory`；报告会保留未绑定标记。

评价包括预测点语义诊断、GT 表面覆盖、类别一致的一对一实例 IoU 匹配、实例 precision/recall、碎裂/误合并，以及指定 `--before` 后的新增点正确率和原有点退化。`--before` 必须是同一几何、同一点序的补全前结果。**这些是明确采样与类别范围的诊断指标，不是官方 ScanNet/3RScan AP。** 报告保留距离阈值、词表范围和输入哈希。

完成建图后，设备页面支持“点云 / 表面”切换。表面是同一 RGB-D 和轨迹的 TSDF 三角网格，使用 RGB 顶点色；预览最多约 15 万个三角面，“导出表面”保存完整 PLY，iPad 使用系统分享。对象语义、实例选择及查询高亮使用点云视图。表面只覆盖实际观测到的区域，不自动补全遮挡或孔洞。

旧的完成记录可在已配置 Open3D 的处理环境补生成表面：`python -m pose_pipeline.surface --session /absolute/path/to/scan_record`。命令校验原输入与轨迹，新增独立表面产物及 `SURFACE.json`，保留原地图和清单；没有表面的旧记录仍可查看点云。新处理记录会自动导出表面。

## 场景图与查询

```bash
revemap build-scene-graph --result outputs/run-stride --output outputs/scene_graph.json
revemap query-scene --graph outputs/scene_graph.json --label chair
revemap query-scene --graph outputs/scene_graph.json --label chair --nearest-to 1
revemap query-scene --graph outputs/scene_graph.json --question '有几把椅子'
revemap query-scene --graph outputs/scene_graph.json --question '桌子附近有哪些椅子'
```

节点包含实例 ID、类别、中心、包围盒及可用的命名支持帧。未知对象和没有匹配项的查询会明确返回 `unknown`。最近对象按中心距离排序，等距时保留并列结果。名称证据是模型观察，不是人工确认标签。

默认建立 `near` 和有向的包围盒 `contains` 关系，查询支持反向的 `inside`。只有已经知道地图坐标系的重力方向时，才可传入 `--world-up X Y Z` 生成 `above` 和 `supported_by`，并查询反向的 `below`。例如仅在确认世界 Z 轴向上时使用 `--world-up 0 0 1`。关系来自包围盒、水平投影重叠和高度间隙，是几何假设；包围盒包含不代表物体真实装载，支撑候选不代表已验证接触。

自然语言入口采用确定的有限语法：查找、当前地图实例计数、附近、上下方、包围盒包含、最近对象以及“且”组合。中英文类别可以匹配已记录的类别或名称。例如 `找出桌子附近且位于显示器下方的物体`。多个参考对象返回 `ambiguous` 和候选实例，选择实例后执行原查询；不自动猜选。未找到参考对象或未确定向上方向时返回 `unknown`，不当作零个。颜色、否定、或条件等未支持语法返回 `unsupported`。计数只表示当前地图中检测到的实例数量。

也可用 `--query-json plan.json` 提交结构化查询。条件均按“候选对象 relation 参考对象”解释，多个条件取交集，最多八个；`nearest` 在其他筛选完成后排序并保留并列结果。

```json
{"schema":"revemap.scene_query.v1","operation":"find","label":"chair","conditions":[{"relation":"near","reference":{"instance_id":1}},{"relation":"below","reference":{"instance_id":2}}]}
```

## 采集界面与恢复

`revemap gui`、`revemap-gui` 现在使用同一设备后端，支持 USB、iPad TCP/可恢复上传、无线 Orbbec 和 RGB-D 回放。iOS 源码在 `ios/`，通过同一后端的内嵌网页查看和查询地图。原 `pose_pipeline.live_gui` 核心处理接口保留兼容；传入设备参数时转到统一入口。用 `revemap gui --help` 查看全部参数。

```bash
revemap gui --runtime configs/semantic_runtime.local.json \
  --provider-root /absolute/path/to/DROID-W \
  --gpu-python /absolute/path/to/gpu-env/bin/python \
  --cpu-python /absolute/path/to/geometry-env/bin/python \
  --capture-python /absolute/path/to/capture-env/bin/python \
  --output /absolute/path/to/scans \
  --library-root /absolute/path/to/previous-scans \
  --replay /absolute/path/to/manifest.json --no-browser
```

实际设备改用 `--ipad-port 7001` 或 `--wireless-host <采集主机IP> --wireless-port 1024`，每个服务只选择一种输入。默认只监听本机；iPad 访问工作站时按已有可信网络配置指定 `--host`。`--library-root` 可重复添加两个设备原来的扫描目录，页面从 `/api/sessions` 统一列出，`/?session=<扫描ID>` 重开记录。原始采集和失败记录保留，每次重新处理写入新的 `session/attempts/<attempt>/pipeline/`。

完成后在“场景查询”中输入问题，命中实例会在同一地图中高亮，支持逐个选择歧义参考物体、查看命名观察裁图、导出物体 PLY 和场景图 JSON。iPad 内嵌视图右上角有“查询”按钮。裁图缺失或内容变化时明确提示不可用。查询与证据绑定扫描和当前处理结果，重跑后拒绝旧 context；历史记录的查询不切换正在采集的会话。计数、类别冲突和未知关系都保留在结果中。小对象的代表点会保留在预览预算内。

HTTP 调用先读 `/s/<扫描ID>/api/status` 的 `scene_context`，再向同一路径的 `/api/query` 提交 `{"question":"有几把椅子","context":"..."}`，或用 `query` 提交上述结构化对象。POST 沿用页面提供的 `X-Scan-Token`；未知向上方向保持 `world_up: null`。`/scene_graph.json` 导出带地图哈希的图，`/api/evidence` 按实例、观察索引和 context 读取验证后的裁图。新结果遵循 `ARTIFACTS.json`；旧记录可读取，但图中保留未绑定来源标记。

GUI 自动记录已完成阶段的检查点，“重新处理”创建新 attempt，并尝试复用上一次通过完整校验的建图和 SAM3 阶段；旧版结果没有检查点时会全量重跑。停止/失败时已保存的完整帧会尽可能封存。缺少彩色帧、深度帧或同步超限都会受到有效帧超时约束。

## 验证与下一步实验

```bash
python -m pytest -q
python -m build
```

CI 覆盖 Python 3.11/3.12 CPU 测试，并在 checkout 外安装 wheel、运行 CLI 和合成查询示例。实测范围、负结果和设备限制见 [本轮验收记录](docs/acceptance-20260922.md)。相机实际拔插、长时间扫描和更广泛的真实数据效果仍需在配置好的设备上继续验收；研究接受条件见 [验证计划](docs/validation.md)。

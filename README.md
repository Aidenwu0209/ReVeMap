# ReVeMap

RGB-D 多视角语义建图、实例恢复与可解释对象查询。原始 RGB-D 用于轨迹估计和几何重建，SAM3 观察经过深度检查、多视角融合及实例关联后写入地图。VLM 名称默认作为观察元数据；可选 refinement 和离线 P2 有独立入口。

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

默认保留原有固定步长选帧。下面的可选模式在最终回填轨迹可用后，用图像清晰度、有效深度比例和位姿差异选取语义帧；**几何重建仍使用全部原始帧**。

```bash
revemap run-sam3 \
  --manifest /absolute/path/to/manifest.json \
  --runtime configs/semantic_runtime.local.json \
  --output outputs/run-diverse \
  --schedule serial --vlm none \
  --view-policy quality-diverse --view-budget 60
```

预算不得超过输入帧数。公平比较时，`stride`、`quality` 和 `quality-diverse` 使用相同的显式 `--view-budget`。视角多样性不足时会按质量补足预算，降级帧记录在 `view_selection/VIEW_PLAN.json` 中；位姿差异不等于统计独立，当前也不代表已经证实准确率提升。

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

## 场景图与查询

```bash
revemap build-scene-graph --result outputs/run-stride --output outputs/scene_graph.json
revemap query-scene --graph outputs/scene_graph.json --label chair
revemap query-scene --graph outputs/scene_graph.json --label chair --nearest-to 1
```

节点包含实例 ID、类别、中心、包围盒及可用的命名支持帧。未知对象和没有匹配项的查询会明确返回 `unknown`。最近对象按中心距离排序，等距时保留并列结果。名称证据是模型观察，不是人工确认标签。

默认只建立 `near` 关系。只有已经知道地图坐标系的重力方向时，才可传入 `--world-up X Y Z` 生成 `above` 和 `supported_by`。例如仅在确认世界 Z 轴向上时使用 `--world-up 0 0 1`。关系来自包围盒、水平投影重叠和高度间隙，是几何假设，不能据此声称验证了真实接触。

## 采集界面与恢复

用 `revemap gui --help` 查看相机、解释器和输出目录参数。界面提供原始颜色、语义与实例查看，历史会话打开，以及使用已封存原始帧重新处理。每次处理写入 `session/attempts/<attempt>/pipeline/`，保留之前的失败记录。

完成后可在“对象查询”中按类别/名称、最近对象或空间关系查找并高亮实例。对象卡显示命名支持帧，可打开仍然存在且哈希匹配的原始观察裁图。查询、裁图及高亮绑定当前地图，切换扫描后旧页签的对象操作会被拒绝。小对象的代表点会保留在预览预算内。

“重新处理”目前是从封存 RGB-D 重新运行完整流程，**不是跳过未核验阶段的断点续算**。停止/失败时已保存的完整帧会尽可能封存。缺少彩色帧、深度帧或同步超限都会受到有效帧超时约束。

## 验证与下一步实验

```bash
python -m pytest -q
python -m build
```

CI 覆盖 Python 3.11/3.12 CPU 测试，并在 checkout 外安装 wheel、运行 CLI 和合成查询示例。真实 GPU 流程、相机断连、长时间扫描和真实数据效果仍需在配置好的设备上验收。研究对照及接受条件见 [验证计划](docs/validation.md)。

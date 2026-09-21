# 本次代码迁移验证

后续勘误（2026-09-15）：本页旧 Mage 调用虽然进程/CUDA 加载成功，但当时适配器漏传图像张量，不能视作视觉推理验收。最新修复与两场 137 张裁剪复测见 [4060 更新](../rtx4060-models-20260915/README.md)。此处保留旧日志和当时记录，不把旧 Unknown 结果用于模型质量排名。

日期：2026-09-15；基线 `741c9656b92953103c4ecdaa2739536dc97c1f42`。本页记录合入代码的兼容性与结果复现，不把历史测评误写成新代码重新完成的全量准确率评测。

## 完整冻结地图重放

新 `build_semantic_bundle.py` 从原始候选掩码、点投影和命名记录构建 bundle；新 `enhance-semantic` 重新计算 T1、P2 和名称确认。没有导入实验目录里的执行脚本，也没有把最终答案当作输入。四场均保持点序、XYZ、原有正语义和原有正实例归属。

| 场景 | 逐点类别名称 | 实例数组 | 置信度 | 补全 / 命名 |
|---|---|---|---|---|
| ScanNet0030 | 与封存结果一致 | 一致 | 一致 | P2 +6,103 点；blackboard 22,637 点 |
| Orbbec | 一致 | 一致 | 一致 | graphics card 123 点 |
| ScanNet0050 | 一致；整数 ID 变化 | 一致 | 一致 | piano 3,957 点 |
| ScanNet0011 | 一致 | 一致 | 一致 | 无新增 |

0050 未分配其他模型才使用的类别 ID，因此比较原始整数会不同；按各自 `classes.json` 转成名称后全部一致。参照是 `mage_joy_vl_20260915_v1/maps/qwen3vl_2b_bf16/`。[逐场核对 JSON](FOUR_SCENE_REPLAY_PARITY.json)。缺失的历史投影从 ssh44 复制到新建 `cache_overlay`，没有修改旧封存目录；前两次因档案缺件而失败的转换也保留在本地整合目录。

## 四次原始 RGB-D 运行

ssh33 新代码、同一模型和相同输入，每场/调度一次；每次重新计算 120 帧轨迹、120 帧几何积分和 25 帧 SAM3。返回码全部为 0，无 GT 输入或单位位姿 fallback。Qwen 只在 SAM3 进程退出后启动。运行标记 `20260915_115720_1188186`。

| 场景 | 调度 | 新实测秒数 | 原始帧 FPS | 地图点数 |
|---|---|---:|---:|---:|
| ScanNet0030 | serial | 111.85 | 1.073 | 40,669 |
| ScanNet0030 | parallel | 102.28 | 1.173 | 40,665 |
| Orbbec | serial | 102.55 | 1.170 | 7,487 |
| Orbbec | parallel | 98.43 | 1.219 | 7,487 |

计时含进程启动、模型加载、推理、几何/语义融合和导出，排除运行前后输入哈希审计。每条件只有一次，未在本次另测峰值显存，不能取代历史重复性能实验；不是全场景、T1/P2 完整流程或在线摄像头 FPS。

每对 25 帧 SAM3 数组全部相同、实例名称元数据相同。新轨迹有小量数值非确定性：0030 地图相差 4 点，双向 5 mm 内点比例 ≥ 99.975%，最近邻语义一致率 99.975%；Orbbec 最近邻语义/实例一致率 100%，双向 5 mm 内点比例 ≥ 99.986%。因此不声称四次地图字节完全相同。这是输出一致性检查，不是 GT 精度验证。[原始指标、事件时间及实际源码哈希](RAW_VALIDATION.json)。

## 单元与几何回归

初次本机批量收集因环境缺少 OpenCV 而中断；随后在本机对其余相关测试执行，125 passed、1 skipped（缺少 Open3D）。缺少环境的 `test_rgbd_measured_refill.py` 两项和 `test_rgbd_refusion_contract.py` 一项转到 ssh33 现有 CPU 环境执行，3/3 通过。没有安装或覆盖依赖。

新增 OCR 隔离测试后 `test_semantic_runtime.py` 为 13/13 通过。累计覆盖 **129 个不同检查**：包括原语义/实例、SAM3 引导与融合、原始建图、进程故障/超时、同帧不能冒充多视角、未知标签保护，以及 API 的图片负载和 OCR 凭证不转发。API 测试使用模拟响应，没有新的云端调用。主 CLI 和轻量 CLI 的帮助入口均能加载。

本机批量使用 `sgf_sga_restore_20260910_v1/analysis_env/bin/python`，`PYTHONPATH=src`；文件集合为 `test_semantic_runtime.py`、`test_sam3_{fusion,guided,loss_unknown,multiview,object_naming,refine,sga,tracking_probe}.py`、`test_semantic_{fusion,mapping}.py`、`test_rgbd_{mapping,mapping_signals,refusion_contract}.py`。远端补测使用 `candidate-v2-runtime-20260904/bin/python`。源码编译检查通过。

## 实际模型调用与独立见证

作者及独立见证各执行一次文档中的 kernel + Qwen/Mage/Joy 两张真实裁剪调用，四阶段均退出 0。三个模型均完整加载到 CUDA，无缺失、错配或错误权重；6/6 裁剪图哈希一致。[独立报告](FRESH_RUNTIME_WITNESS.md)。报告保留所有警告：Mage 自定义代码提示，Qwen/Joy 忽略 `enable_thinking`，Joy 量化慢核回退。`thinking_disabled_verified=false`。

见证输入是已封存的真实裁剪；该旧记录的原始 RGB 绝对路径在 ssh33 不存在，无法在同一主机重新验证从原 RGB 到裁剪的全部变换链。这不影响裁剪实际推理成功，也不能借此声称原始采集链已重新认证。上面的四次原始 RGB-D 运行使用另一份具备完整文件的 120 帧 manifest，运行前后原始 RGB/深度哈希均已核对。

22 个本地配置均完成一张真实裁剪调用，覆盖 Qwen 3.5 / Qwen3-VL / Qwen2.5-VL AWQ、Gemma、MiniCPM-V、InternVL、SmolVLM、Mage、Joy。20 个首次通过；两个 SmolVLM 首次因 `num2words` 缺失而失败，随后复用旧 `pydeps-smol`（num2words 0.5.14、docopt 0.6.2）重新运行通过。旧环境 spec SHA-256 与原档案一致，没有安装依赖。首次失败和修复记录均保留。[22 接口核查](MODEL_INTERFACE_VALIDATION.json)。每次只有一张裁剪，不能据此重新排名精度。

另以 `--vlm none --schedule parallel` 完整重跑 Orbbec120，120 位姿/积分帧和 25 SAM3 帧齐全，VLM 进程未启动、VLM 目录未生成，所有新增名称保持 unknown。91.57 秒 / 1.311 FPS，仅作为一次开关检查；25 帧 SAM3 数组与开 VLM 对照完全相同，最近邻语义一致率 99.987%，差异来自两次新轨迹的数值扰动。[关闭 VLM 的实际检查](NO_VLM_SWITCH_VALIDATION.json)。

未在本轮重新执行 ssh44 的 8B/9B、Qwen2.5-VL BF16 控制组或实际云 API 请求；其接口和历史真实结果均保留。两项 ssh44 小模型 ID 是主机控制别名，对应架构已在 ssh33 测试。API 使用模拟请求测试。

## Mage 无人值守修复

最终还处理了独立见证发现的 Mage `[y/N]` 提示。官方 processor 会丢弃 `trust_remote_code`，所以仅传该参数仍会提示（失败尝试日志保留）；最终传入已经校验、加载的 Mage config，避免 tokenizer 再次询问。关闭标准输入重跑同两张裁剪，退出 0、无交互提示、完整 CUDA 加载，输出仍为 unknown / unknown。[最终检查](MAGE_NONINTERACTIVE_CHECK.json)。其他非阻塞模型警告仍如实保留。

这是四次原始运行后的唯一推理源码变化，限定在 Mage 构造分支；已反向还原改动并核对原 SHA-256，其余 src/config 哈希一致，Qwen 原始管线路径未变。新模块 13 个单元测试再次通过。独立见证报告保留当时原文，不把后续作者复查写成独立见证。

## 产物位置

- 本机：`/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_integration_20260915_v1/`。
- 远端：`/home/aidenwu/Documents/SGF-SGA-experiments/developnew_integration_20260915_v1/tests/`。
- 完整新地图：本机 `replay/`；便携证据：`bundles_v3/`；原始运行拉回件：`remote_raw/`。
- 没有改写旧实验封存件；没有把权重、数据集、API 凭证加入 Git。

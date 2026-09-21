# RTX 4060 模型与调度推荐（2026-09-15）

基于 `developnew@4f0f790` 后封存的模型、参数和原始 pipeline 试验。这次迁移 Mage 图像输入修复，保留 DeepSeek 官方接口及 Qwen 图像预算实验档位；没有修改几何、SAM3、融合阈值或 T1/P2。历史测评与本次代码回归分开记录。

## 两种模式怎么选

| 使用方式 | 首选模型 | 理由及限制 |
|---|---|---|
| 分开运行 `serial`，优先已有完整验证 | `qwen3vl_2b_bf16` | 有同输入串行/并行对照，也有完整冻结地图 T1/P2 的收益验证；8GB 上各 GPU 阶段分开能够运行。不是所有场景准确率第一的证明。 |
| 阶段并行 `parallel`，兼顾命名和显存 | `qwen3vl_2b_nf4` | 本轮固定 18 对象命名 14/18，VLM 阶段整卡峰值 2638 MiB；两场原始 120 帧均完成。属于部署推荐；未证明跨场景精度最优，也没有本轮 NF4 串行配对或 NF4 的 T1/P2 最终地图验收。 |
| 想让两种模式用同一模型，或更重视低显存 | 都显式选 `qwen3vl_2b_nf4` | 串行是可选部署候选；不能给它套用 BF16 的实测串行速度、地图准确率或串并行加速百分比。 |

Gemma4 E2B Q4 可作为备用，图像预算用 **280 tokens**。Orbbec 单次并行 1.221 FPS 略快，但 ScanNet0030 为 1.154 FPS，固定命名分数 7/18；这不支持“换 Gemma 整体更好”。无需因为完全错开就换更大模型，当前证据没有显示更大模型稳定提升。

为兼容旧命令，CLI 省略参数时仍为 `parallel + qwen3vl_2b_bf16 + stride5`；**推荐配置应显式写入命令**。`enhance-semantic` 继续用已验证的 BF16 和对应真实 bundle；不能把模型名改成 NF4 就复用 BF16 的预测证据。

## 可直接使用的两套命令

在仓库根目录执行。先把 [runtime 示例](../../../configs/semantic_runtime.example.json) 复制到仓库外，填写实际解释器、源码、权重及数据路径。保留虚拟环境的 `bin/python` 路径，不要解析符号链接成系统 Python。`python` 可用 CPU 环境解释器，`run-sam3` 自动启动配置指定的各阶段环境。

```bash
# 分开运行：几何 → SAM3 → Qwen BF16 → CPU 语义融合 → 名称回填。
PYTHONPATH=src python -m pose_pipeline.semantic_runtime run-sam3 \
  --manifest /absolute/path/rgbd-manifest.json \
  --runtime /absolute/path/runtime.json \
  --output /absolute/path/new-serial-bf16 \
  --schedule serial --vlm qwen3vl_2b_bf16 --stride 5

# 阶段并行：推荐显式使用 NF4。
PYTHONPATH=src python -m pose_pipeline.semantic_runtime run-sam3 \
  --manifest /absolute/path/rgbd-manifest.json \
  --runtime /absolute/path/runtime.json \
  --output /absolute/path/new-parallel-nf4 \
  --schedule parallel --vlm qwen3vl_2b_nf4 --stride 5

# 在相应 VLM 环境先测同一张真实裁剪；此命令使用当前 Python。
PYTHONPATH=src python -m pose_pipeline.semantic_runtime test-vlm \
  --model qwen3vl_2b_nf4 --runtime /absolute/path/runtime.json \
  --images /absolute/path/object.png --output /absolute/path/new-test-nf4
```

两套输出目录必须不存在。输入 manifest 的 RGB/深度对齐、内参和深度单位按真实设备填写；不能把 ScanNet 的标定复制给 Orbbec。原始几何读取 manifest 内全部帧；不要把实验的 120 帧截断误当部署默认。

## 参数完整设置

### 可配置项及模型生成参数

| 项目 | 分开运行 | 阶段并行 | 设置位置 |
|---|---|---|---|
| 调度 | `serial` | `parallel` | `--schedule` |
| 命名模型 | `qwen3vl_2b_bf16` | `qwen3vl_2b_nf4` | `--vlm` |
| SAM3 采样 | 每 5 帧 + 最后一帧 | 相同 | `--stride 5`；按 manifest 顺序，不是运动自适应 |
| 几何 CPU 线程 | 2 | 2 | runtime 顶层 `threads: 2` |
| 单阶段超时 | 7200 秒 | 7200 秒 | runtime 顶层 `stage_timeout: 7200`；长序列按实际需要增大，不影响精度 |
| VLM CPU 线程 | 4 | 4 | runtime `models.<模型ID>.threads: 4` |
| 主设备 | CUDA 0 | CUDA 0 | 单卡 GPU；不静默降级 CPU |
| 权重 | Qwen3-VL-2B-Instruct | 同一份原始权重，加载时量化 | runtime `models.<模型ID>.weights` |
| 权重版本 | `89644892e4d85e24eaac8bacfd4f463576704203` | 相同 | 注册表和下载收据必须匹配 |
| 权重/计算精度 | BF16 | NF4 权重、BF16 计算、double quant 开 | 注册表 `quant`；NF4 不需要另一份权重 |
| 注意力 | SDPA | SDPA | 适配器固定 |
| 图像面积下限 | 3136 pixels | 3136 pixels | 注册表 `min_pixels` |
| 图像面积上限 | 150528 pixels | 150528 pixels | 注册表 `max_pixels`；不是固定宽高 |
| 输出上限 | 24 tokens | 24 tokens | 注册表 `max_new_tokens` |
| 解码 | `do_sample=False`、`use_cache=True` | 相同 | 适配器固定；贪心解码，不调 temperature/top_p |
| 随机种子 | Torch/NumPy 42 | 相同 | 适配器固定；不保证新轨迹字节一致 |
| 单次图片 batch | 1，逐裁剪执行 | 相同 | worker 固定；不启用 VLM 请求并发 |
| 思考参数 | 请求 `enable_thinking=False` | 相同 | 模板可能忽略；不声称所有后端都确认关闭 |
| 输出要求 | 固定英文物体类别，最多六词，不能确定返回 unknown | 相同 | `common.PROMPT`；不会临时加场景专属提示词 |

路径、`threads`、`stage_timeout` 写入 runtime JSON；图像预算、量化和生成长度来自模型注册表。**在 runtime 里随意添加 `min_pixels`、`IoU` 等字段不会修改这些算法参数。** 已保留 `qwen3vl_2b_bf16_min65536` 实验 ID（65536–602112 pixels），但没有稳定证据支持升为推荐，且它同时改动上下界。

参数试验中，提高 Qwen 输出上限到 64 仍为 11/18；65536–602112 像素候选仅增至 12/18，未过原定筛选。Gemma 官方采样三个种子为 4–8/18，未稳定超过原配置 7/18。因此继续用短输出和贪心解码，不挑选最佳随机种子。

主 VLM 实测环境为 Torch 2.7.1+cu128、Transformers 5.17.0、bitsandbytes 0.50.2；CPU、DROID、SAM3、VLM 使用独立现有环境。完整必填路径、权重收据格式和各家兼容环境见[运行说明](../semantic-runtime-20260915/README.md#配置与依赖)。两种 Qwen 都配置同一个 `vlm_python` 即可；无需为了切换调度重新安装依赖。

Gemma 备用选择 `--vlm gemma4_e2b_qat_q4`：官方 QAT Q4_0 GGUF + mmproj，`image_tokens=280`，不设 image_min_tokens，ctx=2048、parallel slots=1、CPU threads=4、batch/ubatch=256、GPU layers=999、mmproj offload、flash attention 开、fit off、seed=42、temperature=0、top_k=1、max_tokens=24、reasoning off，不缓存 prompt。配置 `models.gemma4_e2b_qat_q4.weights` 和 `llama_server`；这些参数由现有 LlamaNamer 固定执行。

### 两模式共用的几何、语义及实例参数

下面是此次实测沿用的代码值，**不是新调参后的最优解，也不是额外 CLI 开关**。两种模式只换调度/命名器，不通过放松融合门槛制造“更完整”的地图。

| 环节 | 沿用设置 |
|---|---|
| TSDF | voxel=0.02 m，sdf_trunc=0.08 m，depth_trunc=4.5 m；输入全部帧积分 |
| SAM3 | 原始图像输入，BF16 autocast；模型 threshold=0.5；固定 `sam3_indoor_v1.json` 概念提示词 |
| 每像素类别竞争 | 最佳得分 ≥0.5，领先第二类别 ≥0.1；冲突保留 unknown |
| 2D→3D 可见性 | 深度一致误差 ≤0.05 m；使用本次估计位姿、真实深度/内参，不邻域强制填标签 |
| 地图语义投票 | min_views=2，vote_share=0.65，margin=0.15；单次观测且内部像素置信度 ≥0.9 的例外沿用原实现 |
| 初始几何轨迹关联 | min_points=30，min_overlap=0.4；候选差距 <0.15 时拒绝歧义匹配 |
| 对象共识 | ≥2 帧，mean score≥0.8，≥50 点；每侧选择最多 128 个 ≥50 点轨迹 |
| 裁剪选择 | 跳过预测 floor(10)/wall(19)；面积 ≥300 深度像素，bbox 两边 ≥10；面积排序每帧最多 8 个 |
| 裁剪上下文 | bbox 每侧外扩 15%，至少 2 深度像素，按 RGB/深度尺寸换算并裁到图像边界 |
| 名称投影归属 | 对同一实例支持点 ≥30，投影占比 ≥0.65 |
| 名称投票 | ≥2 个不同原始帧同名，且支持帧数严格超过第二候选；不通过保持 unknown |
| 输出 | `fused/export/map_labeled.ply` 含 semantic_id/instance_id；VLM 仅更新 `fused/instance_names.json` |

多视角实例关联全部配置：`min_mask_points=30, min_output_points=50, min_group_frames=2, visibility_fraction=0.3, containment_fraction=0.8, split_piece_fraction=0.2, split_frame_fraction=0.2, min_split_frames=2, min_support_frames=3, consensus_fraction=0.9, cannot_link_frames=2, min_point_views=1, ownership_share=0.65, ownership_margin=0.15, min_object_score=0.8, object_score_mode=max_point`。

引导恢复全部配置：`min_mask_points=30, min_visible_fraction=0.3, containment_fraction=0.8, mask_guide_purity=0.8, split_piece_fraction=0.2, min_split_frames=2, min_support_frames=2, min_point_views=2, min_point_confidence=0.8, min_output_points=50, anchor_min_fraction=0.2, anchor_dominance=0.8`。

几何前端的其余设置继续使用已验证的 [rgbd_droid.py](../../../src/pose_pipeline/rgbd_droid.py)、[rgbd_measured.py](../../../src/pose_pipeline/rgbd_measured.py)、[rgbd_refill.py](../../../src/pose_pipeline/rgbd_refill.py)；不会因切换 VLM 自动重调。完整固定地图 T1/P2 的 bundle 和确认门槛见[原运行说明](../semantic-runtime-20260915/README.md#在固定地图上复现最新改进)，不计入下面速度。

## 真实结果与结论边界

RTX 4060 Laptop 8188 MiB，64GB RAM。下面每个配置/场景/调度仅一次，输入均为 ScanNet0030 或 Orbbec 的前 120 原始帧，25 SAM3 帧；包含新轨迹、新几何、加载、推理、融合、导出，不含前后输入哈希检查。系统文件缓存可能已热，不能证明持续实时或稳定加速。

| 模型 | 0030 serial / parallel 秒 | Orbbec serial / parallel 秒 | VLM 阶段峰值 MiB |
|---|---:|---:|---:|
| Qwen BF16 | 107.85 / 101.67 | 102.20 / 99.76 | 5138 |
| Qwen NF4 | 未在本轮配对 / 103.48 | 未在本轮配对 / 101.19 | 2638 |
| Gemma E2B Q4 | 108.89 / 104.02 | 101.61 / 98.32 | 3616 |

Qwen NF4 并行约 **1.160 / 1.186 FPS**；整条管线峰值约 **6352 MiB**，并没有降到 2638 MiB，因为 SAM3 仍占峰值。显存为 0.25 秒采样整卡值，含桌面/CUDA 上下文；不能相加推算两模型同驻是否可行。

6 组串并行配对（含 Qwen 图像预算候选）均通过原定输出一致性门槛：25 帧 SAM3 数组一致、逐裁剪名称一致、2cm 双向几何覆盖和对应点语义一致率 ≥99.9%。这是预测一致性，不是 GT 正确率。本次实际帧处理/命名请求区间与对应 CPU 工作区间的重叠为零，阶段重叠收益主要来自加载；不是 SAM3 与 VLM 同时在 GPU 推理。旧的 GPU 双模型常驻试验曾 OOM，当前开关避免该路径。

17 个配置（16 个模型/精度 + 1 个图像预算候选），两场共 **42 次执行，40 个有效比较**。原 Mage 两组漏图像张量，仅文字推理，保留但排除；修复后两场重新执行，137 个裁剪均核验有图像输入。对应实际 VLM 修复在本提交合入，不把旧 Unknown 结果解释为 Mage 视觉能力差。

| 同一批 288 张裁剪 | N2 正确 /18 | 秒/裁剪 |
|---|---:|---:|
| Qwen NF4 | 14 | 0.094 |
| Qwen BF16 | 11 | 0.087 |
| Mage NF4（历史正确图像适配器） | 10 | 0.171 |
| DeepSeek V4.1 Flash | 9 | 1.315 |
| Gemma E2B Q4 | 7 | 0.117 |
| JoyAI NF4 | 6 | 0.274 |

这 18 个是 ScanNet0030 固定开发对象；本表是命名分数，受类别字典和投票影响，不是全场景 mIoU。五组本地历史记录重算、DeepSeek 新调用，图片字节一致；API 延迟包含网络。其他场景本轮未完成对应 GT 评分：Orbbec 有名数量 Qwen NF4 4/18、BF16 6/18、DeepSeek 8/18，不能据此说谁正确率最高；ScanNet0050/0011 同样不能这样排名。3RScan 仅 1 个物体、3 张裁剪。

[全部 40 组设备与耗时](PARALLEL_RUNS.json) · [6 组串并行配对](SCHEDULE_PAIRS.json) · [全部配置](TESTED_MODEL_CONFIGS.json) · [逐物体命名与 GT](NAMING_COMPARISON.json) · [分场景命名计数](SCENE_NAMING_COUNTS.json) · [Qwen/Gemma 参数试验](PARAMETER_RUNS.json) · [各参数组完整设置](PARAMETER_ARMS.json)。参数试验属于独立测评，不能与原始 pipeline 秒数合并。

## 保留的 API 接口

`deepseek_v41_flash_api` 使用官方 `https://api.deepseek.com/chat/completions`、服务别名 `deepseek-flash`，从 `DEEPSEEK_API_KEY` 环境变量读取凭证。请求固定 temperature=0、max_tokens=24、thinking disabled、原图 detail=original、stream=false；不发 reasoning_effort，避免与关闭思考的设置冲突。逐张请求，无自动重试、无本地模型 fallback，记录 response_model、fingerprint、finish_reason 和 usage。别名不是不可变权重版本。

```bash
# 在安全环境中预先设置 DEEPSEEK_API_KEY；不要把值写入 runtime 或 Git。
PYTHONPATH=src python -m pose_pipeline.semantic_runtime test-vlm \
  --model deepseek_v41_flash_api --images /absolute/path/object.png \
  --output /absolute/path/new-deepseek-test
```

也可显式用 `run-sam3 --vlm deepseek_v41_flash_api`（需配置运行环境）；该整合路径本次没有云端原始 RGB-D 全流程实测，不给它标完整 FPS。用户选择此 API 会将裁剪发送到该服务。保留原 `glm53flash_api` / `paddleocr_vl16_api`；OCR 仍只作文字证据，不能用作 run-sam3 命名器。

封存 DeepSeek 试验共 425 主样本 +19 重复，444 个 HTTP 成功响应、443 个格式有效结果；449 次 benchmark 尝试中保留 5 个传输失败，另有两次探针。冻结地图回填只更新名字，没有修正 semantic_id。没有发现应替换本地默认模型的地图收益。

## 证据与本次验证

[SOURCE_INDEX.json](SOURCE_INDEX.json) 记录从哪些封存文件逐字节复制及 SHA-256；原始 RGB/PLY/权重/日志留在本机和 ssh33 实验目录，不随 Git 上传。复制的交付文件中 `production_changed=false` 指当时实验，不代表此次整合仍未提交。本次检查见 [VALIDATION.md](VALIDATION.md)。

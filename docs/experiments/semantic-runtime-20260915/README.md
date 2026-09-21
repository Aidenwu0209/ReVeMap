# 语义融合、模型接口与调度整合（2026-09-15）

后续更新见 [4060 两种模式推荐及完整参数](../rtx4060-models-20260915/README.md)。兼容默认仍为 BF16；并行部署建议显式选 NF4。最新分支已修复 Mage 图像编码，并新增 `deepseek_v41_flash_api` 和图像预算候选接口。以下保留本次历史整合的作用范围。

这次把 `741c965` 之后有明确收益的冻结地图实验整理进 `developnew`，同时保留所有实际测试过的视觉模型和 API 接口。默认物体命名器是 **Qwen3-VL-2B BF16**。模型识别、SAM3 分割和 3D 实例融合是不同阶段；更换命名模型并不等于更换分割器。

## 两条可运行路径

| 命令 | 输入与实际工作 | 输出 | 当前边界 |
|---|---|---|---|
| `run-sam3` | 原始 RGB-D → 新轨迹/几何 → SAM3 → 2D/3D 融合 → 可选 VLM 命名 | 带整数语义、实例标签的 PLY，以及实例名称 JSON | VLM 名称写入元数据；不会直接改写 PLY 的 `semantic_id`；不含下面的 T1/P2 |
| `enhance-semantic` | 固定地图 + 已有真实预测证据 → T1 实例归属 → P2 补全 → 多视角确认命名 | 更新语义/实例的 PLY、NPZ、类别字典和对象列表 | CPU 重放现有预测；本命令不重新执行 RGB-D、SAM3 或 VLM 推理 |

因此，完整场景的改进数字来自第二条路径；120 帧的 FPS 来自第一条路径。两者不能拼接成一个“完整最新算法端到端 FPS”。原有 `run-rgbd`、`run-semantic`、SGF 和 SGA 实现仍然保留。新 SAM3 路径使用几何/观测关联，并没有暗中运行 SGF 识别网络或学习式 SGA。

## 采用的改进及条件

1. **T1 锚点与共同可见区域关联。** 每个未知区域使用三个不同原始帧；后续掩码需支持第一视角锚点。通过共同可见点的重叠、排斥关系和多视角所有权，减少错误合并及重复实例。已有正类别和已有实例归属保持不变。
2. **P2 双提示补全。** 对两个新生未知实例分别提示 SAM3；两个独立掩码都需覆盖两个锚点，点集 IoU ≥ 0.7，且至少两个不同原始帧支持。只合并已验证的、不相交实例对，补入原先未分配且语义未知的点；冲突点不分配。
3. **确认后再命名。** 至少两个融合视角的上下文裁剪给出同名，且至少两个融合视角的 SAM3 文本掩码满足置信度、锚点覆盖和参考点集 IoU 门槛，才给整个未知实例写入类别。验证视角不投票，已有类别不被语言模型覆盖。原始置信度保持不变，另存 `naming_support`。

采用记录见 [PROMOTION.json](PROMOTION.json)。116 组参数扫描、全局融合替换、额外表面扩张等没有稳定通过跨场景条件的变体，没有变成默认设置。保留原来的 `741c965` 作为可切回的基线。

## 配置与依赖

从仓库根目录执行，先复制 [配置示例](../../../configs/semantic_runtime.example.json) 到仓库外的本机运行目录，填写绝对路径。路径必须指向实际虚拟环境的 `bin/python`，不要把符号链接解析成系统 Python。

- CPU：NumPy、SciPy、Pillow、plyfile；几何前端还需要其现有 OpenCV/Open3D 等依赖。
- 原始几何：已有 DROID-W provider 和单独 GPU 环境；配置 `provider_root`、`gpu_python`、`cpu_python`。
- SAM3：官方源码、权重、独立环境；权重 SHA-256 必须匹配 `sam3_sha256`。
- 主 VLM 环境实测版本：Torch 2.7.1+cu128、Transformers 5.17.0、bitsandbytes 0.50.2。AWQ 和 MiniCPM-V 4 使用历史兼容环境。SmolVLM 还需要 `num2words==0.5.14` 和 `docopt==0.6.2`；本次复用原有 `pydeps-smol`，只给这两个模型的进程增加该 Python 搜索目录，不改共享环境。
- Gemma 使用带视觉 projector 支持的 llama.cpp CUDA `llama-server`，不是 Transformers；官方 GGUF 和 mmproj 两者都必需。
- 模型文件不随 Git 分发。每个模型目录须有原始下载收据 `DOWNLOAD_COMPLETE.json`，包含 `repo`、`revision`、`files: [{file, sha256}]`。加载前核对仓库版本和每个列出文件的 SHA-256；Mage 另核对已审阅的自定义代码哈希。不能靠改收据冒充另一个模型。

全部模型 ID、固定版本、量化方式、图像预算与后端在 [vlm_models.json](../../../src/revemap/resources/vlm_models.json)。不同精度和图像预算保留独立 ID，方便做受控比较。

## 模型接口测试

```bash
PYTHONPATH=src python -m pose_pipeline.semantic_runtime list-vlm-models

# python 应是所选模型的兼容 VLM 环境解释器。
PYTHONPATH=src python -m pose_pipeline.semantic_runtime test-vlm \
  --model qwen3vl_2b_bf16 --runtime /absolute/path/runtime.json \
  --images /absolute/path/object.png --output /absolute/path/new-test-qwen

# 同一批图片依次测试 Mage / Joy；每次使用新的输出目录。
PYTHONPATH=src python -m pose_pipeline.semantic_runtime test-vlm \
  --model mage_nf4 --runtime /absolute/path/runtime.json \
  --index /absolute/path/crops.json --limit 2 --output /absolute/path/new-test-mage
```

`crops.json` 是数组，例如 `[{"file":"object.png","sha256":"真实文件的64位SHA256"}]`。相对图片路径以索引文件所在目录为基准，也可用 `--image-root` 指定。省略 `sha256` 时会记录当前文件哈希。`test-vlm` 使用当前 Python；切换 AWQ/MiniCPM 兼容环境时也要切换解释器。`run-sam3` 则会自动使用配置中模型专属的 `python`。

输出包括 `INPUTS.json`、`MODEL.json`、`RECORDS.json`、逐请求响应和 `COMPLETE.json`。计时包含加载与逐裁剪时间，不是建图 FPS。`unknown` 是合法拒答，不计为识别正确。缺少模型、加载不完整、输入哈希变化或 API 失败会报错，不会静默换成其他模型。

Gemma 可在测试命令上增加 `--weights` 和 `--llama-server`；普通本地模型也可用 `--weights` 覆盖目录。实际加载版本仍须与注册信息一致。SmolVLM 的 `models.<id>.python` 应指向具备上述依赖的解释器；本项目 ssh33 也提供 `smol-python` 启动器，先加入已有 `pydeps-smol` 搜索路径，再执行同一 `env-fast/bin/python`。不必再次安装依赖。

## 串行、并行和关闭小模型

```bash
# 并行开关：默认 parallel。
PYTHONPATH=src python -m pose_pipeline.semantic_runtime run-sam3 \
  --manifest /absolute/path/rgbd-manifest.json \
  --runtime /absolute/path/runtime.json --output /absolute/path/new-parallel-run \
  --schedule parallel --vlm qwen3vl_2b_bf16 --stride 5

# 完全按阶段串行执行：使用新输出目录并把参数改成
# --schedule serial

# 保留 SAM3，跳过 VLM 加载和命名：
# --vlm none
```

这里的“并行”是**阶段并行**：几何 GPU 工作退出后，SAM3 可与最后的 CPU 几何融合重叠；SAM3 完整退出后，VLM 可与 CPU 语义融合重叠。同一时刻不驻留两个本地大 GPU 模型。`serial` 等待每个阶段完成才开始下一阶段。独立原始帧仍全部参与几何，SAM3 默认每 5 帧处理一帧并保留最后一帧；这是按帧索引取样，尚不是按相机运动自适应取样或在线摄像头流。

原始 manifest 使用现有 `pose_pipeline.contracts.load_manifest` 格式，含场景 ID、相机标定、RGB/深度路径与深度尺度；可参考 [RGB-D 入口说明](https://github.com/Aidenwu0209/SGF-SGAligner/blob/18ddeca303bf9937e855d3ac3113a90dd852a462/docs/depth_first_pipeline_zh.md)。配置和输入哈希随运行记录，子进程失败或超时会终止本次拥有的其他阶段。输出目录必须不存在。

输出位置：

- `mapping/`：本次新建图、轨迹和阶段日志。
- `semantic/frames/`：SAM3 原始掩码、逐点类别缓存及首末帧叠加图。
- `semantic/vlm/`：所选模型的真实裁剪命名记录；`none` 时不产生。
- `fused/export/map_labeled.ply`、`fused/map_labels.npz`：固定 SAM3 类别的语义和实例。
- `fused/instance_names.json`：经过不同帧一致性检查的 VLM 名称元数据。
- `COMPLETE.json`：本次处理帧数、耗时及明确的计时范围。

## 在固定地图上复现最新改进

迁移工具直接读取以前封存的真实掩码和投影，不读取 GT，不复制以前选好的最终答案。它把所有候选掩码转换成便携点集证据，再由新代码重新计算 T1/P2 和命名门槛。

```bash
PYTHONPATH=src python scripts/build_semantic_bundle.py \
  --comparisons-root /absolute/path/comparisons \
  --scene scannet/scene0030_00 \
  --model-run /absolute/path/comparisons/mage_joy_vl_20260915_v1 \
  --output /absolute/path/new-bundle

PYTHONPATH=src python -m pose_pipeline.semantic_runtime enhance-semantic \
  --bundle /absolute/path/new-bundle/BUNDLE.json \
  --surface verified --vlm qwen3vl_2b_bf16 \
  --output /absolute/path/new-enhanced-map
```

缺失的历史投影可放到独立镜像目录，并传 `--cache-overlay`，不必改写旧实验。当前转换器针对本项目 2026-09-14/15 档案布局；它不是任意目录自动识别器。便携 bundle 不依赖这些绝对路径，必须包含点序哈希及所有证据 NPZ 的哈希。`SOURCE_HASHES.json` 是转换时溯源，不是事前注册。

`--surface off` 保留 T1、关闭 P2；`--vlm none` 不给未知实例补类别。选择其他 VLM 时，bundle 必须含该模型真实命名和 SAM3 文本确认结果，缺失就报错。注册一个新模型并不会凭空产生它的多视角确认缓存。

`semantic_labeled.ply` 保存 `semantic_id`、`instance_id`、原始 `confidence` 和 `naming_support`。`classes.json` 是当前输出的类别字典；动态新增类别的整数编号可能因所选模型不同而变化，跨输出比较必须先映射到类别名。XYZ 和点序不变，未知类别可以保留正实例 ID。此 PLY 的 RGB 是语义配色，原始彩色几何仍在输入档案。

## API

GLM 使用 `GLM_API_KEY` 环境变量；PaddleOCR 使用 `PADDLEOCR_API_KEY`。在自己的安全环境中设置，不要把值放到命令历史、runtime JSON、README 或 Git 中。也可在模型配置中用 `token_env` 指定已有环境变量名。

```bash
PYTHONPATH=src python -m pose_pipeline.semantic_runtime test-vlm \
  --model glm53flash_api --images /absolute/path/object.png \
  --output /absolute/path/new-glm-test

PYTHONPATH=src python -m pose_pipeline.semantic_runtime test-vlm \
  --model paddleocr_vl16_api --images /absolute/path/object.png \
  --output /absolute/path/new-ocr-test
```

显式选择 API 会发送这些裁剪图到所选服务。PaddleOCR-VL-1.6 返回文字证据，不能作为物体分割或命名器直接替换 SAM3/Qwen，因此它只保留在 `test-vlm`，不能选作 `run-sam3 --vlm`。等待 OCR 队列超时会报错，服务端任务可能仍在队列中。

## 结果与证据

根目录 [README](../../archive/developnew_README.md) 展示主要结果及全部模型表。原始汇总的可读数据保存在：

- [MODEL_RESULTS.json](MODEL_RESULTS.json)：29 行历史模型测评（含控制重复），每行 288 次真实裁剪调用。
- [MAP_GT_RESULTS.json](MAP_GT_RESULTS.json)：固定 ScanNet0030 82 对象诊断与 GT 点级核对。
- [RAW_TIMINGS.json](RAW_TIMINGS.json)：历史 120 帧串行/并行原型的 22 个完成运行及成对一致性检查。
- [NO_VLM_RESULTS.json](NO_VLM_RESULTS.json)：独立配对的小模型开/关消融，8 次运行。
- [PADDLE_RESULTS.json](PADDLE_RESULTS.json)：18 张图片 OCR 结果和融合前后比较。
- [SOURCE_INDEX.json](SOURCE_INDEX.json)：汇总来自哪个封存文件及其 SHA-256；8 个源文件均与原交付封存哈希一致。
- [VALIDATION.md](VALIDATION.md)：本次代码迁移后的验证；区别于上面的历史效果与性能测试。

全部原始 PLY、预测和日志继续保存在本机 `SGF-SGA/comparisons/` 对应实验目录，不上传权重、数据集和 API 凭证。相关目录为 `semantic_identity_20260914_v1`、`semantic_surface_support_20260914_v1`、`small_vlm_benchmark_20260914_v1`、`model_scale_deployment_20260915_v1`、`no_vlm_ablation_20260915_v1`、`mage_joy_vl_20260915_v1`。本次新验证放在 `developnew_integration_20260915_v1`。

## 尚未解决的问题

- 这批对象和场景用于开发选择，不能代替未见场景泛化测试。Orbbec 缺少完整实例 GT，显卡命名成功也仍只有 123 个地图点。
- 3RScan 本轮只有小范围命名样本，没有完整的 T1/P2 地图或原始流水线验收。
- 大模型实际运行在 ssh44，未能登录 ssh18；不要把 ssh44 结果写成 ssh18 结果。ssh44 的 Windows/WSL 计时存在时钟偏差，不能直接跨机器比较毫秒延迟。
- 原始管线仍为离线文件输入，尚未把完整 T1/P2 证据生成接成单条自动原始流命令，也没有测得它的完整 FPS。
- 模型类别词可能比 GT 粗或细。Mage 的 `chalkboard` 与 GT `blackboard` 在冻结严格词表中不相等，这种扣分不能证明它视觉上完全没认出黑板。
- Mage 自定义类、Joy 量化慢核、部分模型忽略 `enable_thinking` 的警告保留在日志。记录 `thinking_disabled_verified=false`，不声称已统一验证关闭思考模式。

# 运行中切换对象命名模型

在扫描设置的“命名模型”中选择 Qwen3-VL-2B NF4（原版）或 Empero Qwen3.8-Distill 4B Q8 · 34 GPU。扫描过程中也可以切换。正在推理的裁图由原模型完成，下一张待命名裁图使用新模型。每次切换先释放旧模型，再按需加载新模型。已完成的名称保留。要让整段录制统一使用某个模型，可以用该模型重新处理已有录制。

| 配置 ID | 语言推理 | 视觉推理 |
|---|---|---|
| `qwen3vl_2b_nf4` | 原版 Transformers NF4 | 原版视觉编码器 |
| `qwen38d_4b_q8_gpu34` | llama.cpp Q8_0，34/34 层 CUDA | 同检查点的 F16 投影器，CUDA0 |

4B 会在启动时核对两份权重的 SHA-256、实际 GPU 层数和视觉后端。它需要全 GPU 部署的显存余量。在一台 16 GiB RTX 5000 Ada Laptop 上，桌面等进程已占约 2.60 GiB 时，单独运行此命名服务的整卡采样峰值为 8.11 GiB。

## 配置新工作站

1. 沿用原有 2B 配置和 VLM Python 环境。4B 需要支持 CUDA 及多模态的 llama.cpp b12676。安装好该版本的服务和配套动态库。
2. 从固定版本获取语言权重、原始视觉检查点并构建投影器。在含有 llama.cpp 转换依赖且已安装 ReVeMap 的 Python 环境中，从仓库根目录运行以下命令。转换器对应提交为 `d35dc4d9e6827db8e08426a2340d8a71ea863cd9`。

```bash
hf download empero-ai/Qwen3.8-4B-Distill-GGUF Qwen3.8-4B-Q8_0.gguf \
  --revision 391fc7d103e3942a408def3e4f51c2f85d464417 --local-dir downloads/empero-language
hf download empero-ai/Qwen3.8-4B-Distill config.json preprocessor_config.json model.safetensors \
  --revision c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e --local-dir downloads/empero-checkpoint
python tools/prepare_empero_4b.py \
  --language downloads/empero-language/Qwen3.8-4B-Q8_0.gguf \
  --checkpoint downloads/empero-checkpoint \
  --converter /path/to/llama.cpp/convert_hf_to_gguf.py \
  --output /path/to/models/qwen38d_4b
```

3. 在本机 runtime JSON 的 `models.qwen38d_4b_q8_gpu34` 中填写 `weights` 和 `llama_server` 的绝对路径，可参考[配置示例](../configs/semantic_runtime.example.json)。需要额外 CUDA 库时，在该模型配置的 `library_dirs` 列表中填写目录。GUI 热切换的两个模型共用同一 VLM Python 环境。
4. 如只需这两个界面选项，可在 runtime JSON 顶层设置 `"gui_vlm_models": ["qwen3vl_2b_nf4", "qwen38d_4b_q8_gpu34"]`。重启 GUI 以加载本次代码更新。后续切换在界面上完成。

## API 与验证

`GET /api/models` 返回已配置的模型 ID、显示名和可热切换的 ID。用扫描页面提供的 `X-Scan-Token` 向 `POST /api/options` 发送 `{"vlm":"qwen38d_4b_q8_gpu34"}`，或者发送原版 ID 切回。成功响应确认所选配置。实际推理使用的模型记录在每张裁图的 `model_id` 中，每次加载的审计文件列在 `MODEL_LOADS.json` 中。两种配置同样适用于 GUI 的 refinement 命名阶段；缓存按模型和裁图内容区分。

也可通过 CLI 在启动一次处理时指定 `--vlm`，或用一张真实裁图验证 4B 安装：

```bash
python -m pose_pipeline.semantic_runtime test-vlm --model qwen38d_4b_q8_gpu34 \
  --runtime configs/semantic_runtime.local.json --images /path/to/crop.png --output /path/to/new-test
```

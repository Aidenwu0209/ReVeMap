# 独立 runtime 文档见证

结论：本次文档命令的调用兼容性通过，警告与证据边界如下。没有开展新的准确率、FPS 或完整 RGB-D 流水线测试。

- 独立见证代理：`semantic_runtime_doc_witness`；运行标记：`20260915_115144_1179369`。
- 已先读取 `run-experiment/SKILL.md`、`shared-references/compute-env-contract.md` 和本目录 `.aris/compute/ssh33.md`。以下命令逐字执行 **一次**，没有修改代码、配置、模型或环境，没有安装依赖、补丁修复或扩展模型试验。

```sh
ssh -T -o BatchMode=yes -o ConnectTimeout=10 100.72.138.33 /home/aidenwu/Documents/SGF-SGA-experiments/small_vlm_benchmark_20260914_v1/env-fast/bin/python /home/aidenwu/Documents/SGF-SGA-experiments/developnew_integration_20260915_v1/validate_remote.py --witness
```

- SSH 退出码 `0`；`kernel`、`qwen3vl_2b_bf16`、`mage_nf4`、`joy_nf4` 的返回码全部为 `0`。
- 终端输出 `VALIDATION_COMPLETE /home/aidenwu/Documents/SGF-SGA-experiments/developnew_integration_20260915_v1/tests/20260915_115144_1179369 True`；该目录的 `COMPLETE.json` 为 `all_success: true`。
- 种子为 42 的 CUDA 矩阵运算输出 `CUDA_WITNESS [16, 16] NVIDIA GeForce RTX 4060 Laptop GPU`。
- 环境版本经只读 metadata 检查：Torch `2.7.1+cu128`、Transformers `5.17.0`、bitsandbytes `0.50.2`，与文档一致。未重建环境；本见证不单独认证文档中的环境 spec hash。

已逐项检查三个 `MODEL.json`、三个模型 `COMPLETE.json`、全部六条 `RECORDS.json` 记录及四份日志：

| 模型 | 记录的仓库与 revision | 实际模型类 / 精度 | 完整加载证据 | frame 165 / 145 输出 |
|---|---|---|---|---|
| qwen3vl_2b_bf16 | Qwen/Qwen3-VL-2B-Instruct @ `89644892e4d85e24eaac8bacfd4f463576704203` | Qwen3VLForConditionalGeneration / bf16 | 625/625 权重；11 个 verified files | blackboard / blackboard |
| mage_nf4 | microsoft/Mage-VL @ `d88b153285f1633a61b2f693c59c8576693af185` | MageVLForConditionalGeneration / nf4 | 696/696 权重；22 个 verified files | unknown / unknown |
| joy_nf4 | jdopensource/JoyAI-VL-Interaction @ `86124c620faddd7be7b7f2722f29cd5192f21c38` | Qwen3VLForConditionalGeneration / nf4 | 750/750 权重；17 个 verified files | chalkboard / blackboard |

三个模型的 `missing_keys`、`unexpected_keys`、`mismatched_keys`、`error_msgs` 均为空；`parameter_devices` 均只有 `cuda:0`，GPU 名称匹配。`runtime.json` 分别指向原有 `mage_joy_vl_20260915_v1/models/{qwen3vl_2b,mage,joy}`；身份收据和加载记录没有显示模型替换。Joy 使用 Qwen3VL 类，但仓库身份记录仍为 Joy，不能仅凭类名认定替换。三个完成文件均报告 `crops_executed: 2`；六条记录均 `executed: true`、`valid: true`、`output_tokens: 3`。Mage 的 unknown 是实际输出，不是分类正确性的证明。

输入均为 `scannet/scene0030_00`、instance 101 的 frame 165 和 145 裁剪图。远端 PNG 与记录的 SHA256 全部一致，尺寸分别为 640×324、640×342；本地对应源裁剪的 SHA256 也一致，已目视确认两张均为真实黑板场景照片。

- frame 165：`d06a68628c4d04c284aac6ddda964605581add0cd47bc89b8d7b0266a1e6b231`。
- frame 145：`39e4be0d7ff9ba0a4cffd871a63d468a38d6488d4131446c1c457bdace960ca5`。

文档与实况差异、警告和限制：

1. 命令路径、参数、四阶段成功和唯一输出目录均符合文档，无需临场补参数。预检 GPU 为 870/8188 MiB，超过通用 skill 的 `<500 MiB` 空闲定义；compute-apps 仅列 `gnome-remote-desktop-daemon`（96 MiB），无模型计算任务，与本机文档“Desktop remains active”相符。已按授权运行并披露这一预检差异。
2. Mage 日志有 6 条 `[ERROR] Config not found for mage`，并出现 custom-code 信任询问 `[y/N]` 及 `mage_vl` 实例化为空 model type 的警告。没有人工输入、重试或修复，进程仍自行完成；因此这是成功但并非无警告的运行，不能描述为静默加载。
3. Qwen **和** Joy 都报告 `processor_kwargs` 传参方式警告，以及 `enable_thinking` 无效、被忽略。文档只特别提及 Joy；三个 `MODEL.json` 均为 `requested_thinking_disabled: true`、`thinking_disabled_verified: false`，不能宣称已验证关闭思考模式。
4. Joy 的 bitsandbytes 报 inner dimension 4304 不符合 blocksize 64 的快速核对齐条件，回退到较慢实现。该警告与文档预告相符；本次时间不能充当新 FPS 比较。
5. 记录的原始 RGB 路径 `/home/aidenwu/Documents/sgf_sga_all_scannet_orbbec_20260910_v1/inputs/scannet/scene0030_00/color/{165,145}.png` 在 ssh33 均不存在。额外只读溯源检查先遇到 `FileNotFoundError`，随后明确检查到两条路径缺失；没有复制或修复。裁剪图可用且哈希匹配，但原始 RGB 哈希和原始图到裁剪的变换链未能在该主机复核。此检查失败不属于模型命令运行失败。

GPU 工作已结束；退出后 compute-apps 仍仅列上述桌面守护进程，无本次模型进程。此报告是本次独立审阅的唯一手工写入文件；原始见证产物保留在上述远端运行目录。

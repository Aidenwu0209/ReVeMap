# ReVeMap

**Multi-View Verified Instance Recovery for RGB-D Semantic Mapping**

**面向 RGB-D 语义建图的多视角验证实例恢复方法**

ReVeMap 从 `SGF-SGAligner` 的 `developnew` 分支独立抽取，保留 RGB-D 建图、
SAM3 语义/实例融合、多视角实例恢复、VLM 命名、未知点优化和扫描 GUI。
原仓库和原分支保持不变。来源提交及逐文件哈希见
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)。

## 安装与使用

```bash
git clone https://github.com/Aidenwu0209/ReVeMap.git
cd ReVeMap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
revemap --help
revemap list-vlm-models
python -m pytest -q
```

上述安装用于 CPU 数据处理、命令入口和回归测试。完整 RGB-D 建图需要 DROID-W
运行环境，SAM3 和 VLM 需要各自的模型环境与权重；这些不随仓库发布。
模型推理继续按阶段隔离运行，避免把所有模型依赖混装在一个环境中。

将 [配置示例](configs/semantic_runtime.example.json) 复制成
`configs/runtime.local.json`，填写本机真实路径。详见 [环境与输入](docs/SETUP.md)。

| 命令 | 用途 |
|---|---|
| `revemap run-rgbd` | 原始 RGB-D → 估计轨迹与几何 |
| `revemap run-sam3` | 新建地图 → SAM3 分割 → 多视角融合/恢复 → 可选名称元数据 |
| `revemap enhance-semantic` | 在已有真实证据 bundle 上重放 T1/P2 和确认命名 |
| `revemap refine-semantic` | 对已有地图选择视角，重新推理并补充有证据支持的未知标签 |
| `revemap gui` | 相机采集或录制数据回放、几何预览、停止后语义建图与导出 |

新入口使用 `revemap`；原 `pose_pipeline.semantic_runtime` 包路径保留兼容。
新的主线不需要旧 SGF 识别网络、学习式 SGAligner 或 GeoTransformer。
扫描界面使用方法见 [GUI 文档](docs/gui/README.md)。

## 方法与结果边界

`run-sam3` 与冻结证据的 `enhance-semantic` 是两条不同路径；后者的完整 T1/P2
证据生成尚未自动接入前者。`run-sam3` 的 VLM 名称写入元数据，确认增强和
refinement 则可以更新 `semantic_id`。固定几何处理不产生未观测表面，也不能
修复错误轨迹或几何折叠。GUI 在采集停止后生成语义结果，不宣称实时语义推理。

[历史实验目录](docs/experiments/README.md) 保留真实开发结果与失败边界。
仓库独立化没有重新运行这些 GPU 实验，不表示发布了论文、取得新的指标，或
完成全数据集验收。当前迁移验证见 [VERIFICATION.md](docs/VERIFICATION.md)。

## 来源与许可

源分支：`Aidenwu0209/SGF-SGAligner:developnew`

源提交：`18ddeca303bf9937e855d3ac3113a90dd852a462`

新仓库采用独立初始提交，旧开发历史仍在原仓库中保存。保留原 MIT 版权声明，
外部模型、SDK 和数据遵守各自许可。详见 [来源与拆分说明](docs/MIGRATION.md)、
[NOTICE.md](NOTICE.md) 和 [LICENSE](LICENSE)。

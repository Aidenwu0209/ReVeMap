# 本次合入验证

基线：`4f0f790e90deb899e1b3e656ddd19aef5e599fd5`，2026-09-15。此次是已完成实验的修复/接口/说明迁移，没有新增 GPU 性能跑分或付费 API 调用。原始性能、精度和 42 次运行属于独立封存实验。

- **89 个回归检查全部通过**：semantic_runtime、SAM3 投票/引导恢复/多视角融合、semantic_fusion。测试环境为现有 `sgf_sga_restore_20260910_v1/analysis_env`，没有安装新依赖。
- Mage 实际适配器的图像输入测试：必须显式把图片交给 processor；缺失像素或图像 token 数与网格不符时在生成前失败。测试使用轻量张量替身，无 GPU 权重加载。
- 整个 `TransformersNamer` 的 AST 与已在 ssh33 两场、137 张裁剪执行的修复快照完全一致；不能把这次 AST 校验再计为一次 GPU 实验。
- DeepSeek 使用模拟 HTTP 检查完整负载：原图 base64、detail=original、24 tokens、thinking disabled、无 reasoning_effort；401/402/500 均报错，不返回伪造名称。检查 fingerprint/finish_reason 记录，GLM 既有 payload 保持。
- 已有模型注册项全部保持；17 个并行实测配置与封存注册项一致，仅新增一个预算候选和一个 API ID。CLI 的默认模型/调度保持，新增 ID 可以从 test-vlm 和 run-sam3 选择。
- 原几何、SAM3、语义融合、调度与 T1/P2 文件未变。本次没有凭新命名评分修改地图算法。
- 10 份封存证据逐字节复制，全部 SHA 与 SOURCE_INDEX 一致；分场景计数是从这些记录派生，不是新 GT 评测。

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_semantic_runtime.py tests/test_sam3_fusion.py \
  tests/test_sam3_guided.py tests/test_sam3_multiview.py tests/test_semantic_fusion.py

PYTHONPATH=src python -m pose_pipeline.semantic_runtime run-sam3 --help
PYTHONPATH=src python -m pose_pipeline.semantic_runtime test-vlm --help
```

[本次检查记录](INTEGRATION_CHECKS.json) · [封存来源与哈希](SOURCE_INDEX.json)。README 的模型建议是基于现有证据的部署取舍；不称为所有场景最优参数，不把 NF4 命名 14/18 当作地图 mIoU。

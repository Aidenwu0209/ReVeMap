# 融合与补全优化

本次从本地研究提交 `417e122` 选择四项已验证改动，基于 main `04ee3a4` 集成：补全判定与接口修复、候选入口及扩展、保守实例碎片合并、合并步骤精确加速。默认实例策略仍为 `legacy`；`verified` 显式启用碎片合并与额外候选扩展。轨迹、图优化、深度处理及 TSDF 沿用 main。

## 启用方式

在已有模型运行环境上，用新的输出目录运行：

```bash
python -m pose_pipeline.live_semantic \
  --manifest /path/to/manifest.json \
  --runtime /path/to/runtime.json \
  --output /path/to/new-output \
  --instance-policy verified --refine
```

`revemap run-sam3` 也接受 `--instance-policy verified`，执行融合并保存候选。`--refine` 属于上面的 live bridge；也可使用 `revemap refine-semantic` 处理已准备的补全工作区。省略 `--instance-policy` 时使用 `legacy`。合入代码不会自动改变服务器部署、GUI 启动参数或已有地图。

## 实例合并与加速

`verified` 在原 guided recovery 后，合并类别唯一且相同的实例碎片。要求至少三个共享主导 mask 的帧、至少 80% 包含、90% 一致、每帧至少 30 个可见点，并拒绝有至少两帧分离证据的候选。组件合并重新检查联合可见证据与跨组件分离证据，保存 `FRAGMENT_MERGE.json`。几何、语义和已分配点集合不变。

实现用一次分组和稀疏批量计算复用单实例证据，跳过异类组合。加速保持原合并规则和审计结果；它不拆分已有错误实例，也不纠正已有错误类别。

## 补全候选和判定

融合单独保存 `object_geometry.npz`、`candidates.npz` 和带输入哈希的 `CANDIDATES.json`。候选必须未知、未归属，并有至少两个深度一致视角和至少 50 个对象候选点；只作为命名与 grounding 的临时锚点。live bridge 核对候选与基础标签的哈希，再将候选纳入补全输入锁。

`verified` 还能在原混类几何组内寻找纯 mask 的局部证据。mask 必须通过原 multiview 过滤和组评分，包含至少 30 个同类同 owner 已知锚点；正票来自该点原获胜组，其他 eligible 组的异 owner 票否决。每点至少两个不同帧、每 owner 至少 50 个新增候选，floor/wall 不作为目标。原候选和正式标签保留。

最终赋值仍需原 VLM 共识、SAM3 质量门槛和逐点多帧支持。只确认直接观察到的点；已有语义、已有实例归属和 confidence 保留。补全现在先筛合格 mask 再排序、保留 unknown 占多数时唯一的已知类别、跳过 assignment 必然拒绝的异类查询；无候选时跳过模型加载。精确拼写修复将词表的 `refridgerator` 规范化为 `refrigerator`，保留 raw/fine 名称和类别 ID。

## 已有验证及边界

十场固定缓存、相同坐标和实例编码下，用户原 PQ 脚本的场景等权均值：

| 阶段 | 平均 PQ | 证据范围 |
|---|---:|---|
| legacy | 25.7143% | 原融合标签逐点复现 |
| verified 碎片合并 | 26.1887% | 4 场提高，6 场持平 |
| verified 加严格补全 | 26.2599% | scene0070 新增 298 个 chair 点，其余 9 场最终标签不变 |

新增点按用户脚本的 GT 最近点映射属于同一椅子实例；原实例中的地面污染仍存在。十场已用于多轮开发，不能视为独立留出集。固定几何未改变，没有新增 CD 或轨迹改善结论。预测先封存，再读取 GT 评测；用户脚本和坐标约定未变。

合并步骤十场三次中位耗时之和为 10.6437 s → 4.5657 s，减少 57.10%，标签与完整合并审计一致。这是缓存加载后的单个步骤；第一轮完整缓存融合曾增加 13.97% 耗时，因此不能宣称整条 pipeline 加速 57.10%。拼写和判定小修未单独取得真实 PQ 收益。

[逐场指标与耗时](../verification/fusion-refinement-20261001/per_scene.csv)及[选择来源与源码哈希](../verification/fusion-refinement-20261001/SOURCE_SELECTION.json)记录精确来源。完整历史证据仍保留在研究工作区和远端 `/home/aidenwu/Documents/ReVeMap-main-quality-20261001`；本次没有重跑 GPU 或完整重建。

`tools/replay_instance_quality.py` 可在原封存 mask 和投影上验证 legacy/verified；`tools/replay_refinement_quality.py` 通过实际 live bridge 重放封存候选，需原 VLM/SAM3 运行环境。工具要求新输出目录，保留原始输入。

四场真实原生缓存回放完成。兼容性通过；主 IoU≥.5 无匹配数收益，补充 IoU≥.25 的 0200 增加一个 class-agnostic 匹配，全四场纯度下降。整体生产质量改善依据仍不足；主仓不合入，默认保持 off。

顺序固定为 0062→0097→0347→0200；后三场来自见质量前的预选清单，没有筛场或调参。0062 验过 control/default/off/on，后三场只做 control/on。所有预测复用完整验证的原生多类与 Qwen 缓存，模型调用和 GPU 均为零。control 为当前 HEAD 加双方相同的显式资源环境修订，原源码副本保留；未宣称执行文件与原 HEAD 完全逐字节相同。

复用既有 evaluation.py：5cm、first finite paired pose、不做 ICP/scale；主 IoU≥.5、补充≥.25、split/merge fraction=.1。Class-agnostic 分母包含全部 GT 对象，class-aware 分母是精确同名词表内 GT 对象，所有预测实例仍进入 precision 分母。GT 仅在预测封存后由独立 evaluator 读取。以下是 historical development 诊断，不是 official AP、开放语言或独立泛化。

| 场景 | 点数 | 新增 | agn .5 | agn .25 | aware .5 | aware .25 | correct/add .5 | correct/add .25 | outside instance | outside class | purity control→on |
|---|---:|---:|---|---|---|---|---|---|---:|---:|---|
| scene0062_00 | 88879 | 95 | 5→5/24 | 8→8/24 | 4→4/10 | 7→7/10 | 11/95 | 12/95 | 43 | 48 | 0.914135642→0.913421434 |
| scene0097_00 | 105781 | 371 | 8→8/23 | 11→11/23 | 7→7/10 | 8→8/10 | 51/371 | 51/371 | 1 | 335 | 0.949807894→0.948776679 |
| scene0347_00 | 94313 | 481 | 6→6/28 | 10→10/28 | 4→4/11 | 6→6/11 | 31/481 | 35/481 | 71 | 368 | 0.920316967→0.917512231 |
| scene0200_00 | 128056 | 2393 | 8→8/19 | 11→12/19 | 8→8/13 | 9→9/13 | 183/2393 | 1610/2393 | 209 | 1630 | 0.921179542→0.919142389 |

correct/add 定义为本档 class-agnostic 一对一 IoU matcher 已匹配 owner 在 5cm GT-instance support 内的新增点；它不是不受匹配阈值约束的一般逐点 GT 精度。outside instance/class 也必须与各自支持域一同解释。原有 semantic/confidence、正 owner IDs 和实例集合保持不变；所有新增点 semantic 为 unknown，未伪造类别。完整 support 更新 point_count/bbox/空间关系，原类别、名字和 crop 证据逐项不变。

每场原 base 与 control 数组精确一致，geometry contract 重算通过；export 属性除允许的 instance_id 外一致，point lineage 精确覆盖新增点。所有场景 strict artifact、scene graph 和实际 GUI Controller.scene_graph 加载/缓存/查询通过。GUI 工作流和后续 refine 不在此运行范围；opt-in+refine 明确拒绝，off/refine 行为保持。

全部两档匹配数与匹配得失、同一匹配的 IoU 变化、split/merge、before/after、原 JSON 哈希、实际 UTC、原生命名来源、源与输入鲜核及输出 seal 在相邻 JSON 中完整保存。0200 在补充门出现 pred28→GT11 的新增 class-agnostic 匹配（IoU=.368567104），class-aware 数不变；因此新增 threshold-matched-owner correct 从主门 183/2393 到补充门 1610/2393，这个跳变受匹配门定义影响，不是无条件逐点精度的跳变。该真实正结果与四场纯度小幅下降、小幅 IoU 混合变化一并保留，不改变预定主门或用替代指标转正。wall 是实际功能回放时间，可能有共享主任务，不作为无争用性能比较；数值环境无人工线程 cap，但保留原 cKDTree workers=1，不保证持续满核。

合入依据尚不足。可以保留此隔离 opt-in 接入作为研究工件；当前四场结果不支持将其作为整体质量提升写入生产默认或主仓。

0200 真实支持图候选为 pred28→GT11：原预测类别 semantic ID 9、原名称 `door`（实例自由名字为 null），GT 原始 objectId=10，对应 evaluator GT ID 11，原标签为 `closet doors`。原/新 reference-vertex IoU 为 0.09218916046758767→0.36856710393541875；intersection 为 347→1461 个 GT 顶点，projected-owner support 为 429→1743，GT 对象固定 3682 顶点，union 为 3764→3964。该 GT 原标签不在当前精确同名词表，因此收益只进入 class-agnostic 支路。原有 306 个 `door` 语义点保留，新加入该 owner 的 1613 点均为 unknown；未把这些点伪造为 door。它可作为未来真实支持图候选，不改变主阈值或总体质量口径。完整输入/GT 哈希和原字段在 `scene0200_00/SCENE0200_OWNER28_GT11_DIAGNOSTIC.json`。

"""Build the source-backed closeout report and local results gallery."""
from pathlib import Path
import csv
import html
import json
import markdown

R = Path(__file__).resolve().parent
STAGES = {'guided_recovery_maskveto': '三维引导恢复', 'unknown_contract': '实例与类别分离', 'raw_unknown': '原始掩码去重＋类别分离'}


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return f'{100 * value:.2f}%'


def main():
    jobs = read(R / 'inputs/JOBS.json')
    rows = []
    for job in jobs:
        key = job['key']
        c = read(R.parent / 'sam3_sga_20260912_v1/objects_geometry' / key / 'result.json')
        previous = read(R.parent / 'sam3_multiview_20260912_v1/consensus_matched' / key / 'result.json')
        row = {'scene': key, 'frames': len(job['selected_frame_ids']), 'C': c['instance_coverage'],
               'R3': previous['instance_coverage'], 'semantic_coverage_unchanged': c['semantic_coverage']}
        for stage in STAGES:
            result = read(R / stage / key / 'result.json')
            assert result['status'] == 'completed' and result['processed_frames'] == row['frames']
            row[stage] = result['instance_coverage']
        rows.append(row)
    with (R / 'SUMMARY.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    legacy = {'C': read(R.parent / 'sam3_sga_20260912_v1/INSTANCE_0030_objects_geometry.json'),
              'R3': read(R.parent / 'sam3_multiview_20260912_v1/INSTANCE_0030_consensus_matched.json')}
    for stage in STAGES:
        legacy[stage] = read(R / f'INSTANCE_0030_{stage}.json')
    named = {stage: read(R / f'NAMED_CLASSAGNOSTIC_0030_{stage}.json') for stage in ('unknown_contract', 'raw_unknown')}
    loss = read(R / 'loss_audit/orbbec/scan_20260909_142829_5ef1fa/backend_loss.json')
    naming = read(R / 'NAMING_SUMMARY.json')
    tracking = read(R / 'tracking/SUMMARY.json')
    summary = {'quality_accepted': False, 'promoted_to_developnew': False,
               'source_branch': 'experiment/sam3-guided-20260912', 'parent_commit': 'f99f359',
               'reference_C_commit': '08f8754', 'developnew_unchanged': '0c94449',
               'same_map_cpu_arms': 3, 'final_cpu_maps': 15, 'explicit_naming_runs': 10,
               'same_map_rows': rows, 'legacy_instance_0030': legacy,
               'named_instance_0030': named, 'tracking': tracking,
               'scope': 'five cached scenes; 3RScan six frames only; 102-frame paired single-concept GPU probe plus blue-device single-frame probe',
               'main_finding': 'Orbbec has both rejected existing instance groups and missing effective custom-object prompt candidates; a single universal replacement has not passed.'}
    (R / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = [
        '# SAM3 多方向验证结果 · 2026-09-12', '',
        '本轮有明确诊断进展，但没有找到在三个数据集上均优于旧 C 的统一候选。保持 `developnew@0c94449` 不变，全部改动与结果封存在独立实验分支。最值得继续的方向是补充能覆盖陌生设备的描述性候选，再用多视角证据验证；三维引导可做局部修复，纯视频替换尚不通过。', '',
        '入口：[全部效果图和 PLY](index.html) · [视频/提示实验详情](tracking/REPORT_zh.md) · [数字汇总](SUMMARY.csv) · [输出校验](FINAL_OUTPUT_AUDIT.json)。', '',
        '## 1. 实际完成的验证', '',
        '- 同一原始地图、点顺序、冻结轨迹与选帧：三维引导恢复、实例与类别分离、原始掩码去重三组，共 15 张实验地图。五场为 ScanNet0030/0050/0011、Orbbec 4812 帧场景、一个 3RScan 场景的六帧缓存。',
        '- 五场逐点复现 R3 实例输出，审计每个旧实例丢失点到达的最后阶段。',
        '- 对两个类别分离分支增加显式多视角对象命名，共 10 次；只新增 `objects_named.json`，原始逐点语义、实例、地图、`objects.json` 不变。',
        '- ssh44 / RTX 5070 Ti 实际运行 SAM3：四个片段共 102 个观测的图像/视频配对，另做卷盘视觉提示与蓝色设备单帧四条件对照。',
        '- 90 项相关测试通过，15 张地图的源顶点字段、标签、对象清单和输出契约通过核验。运行器、模型、输入和传回文件保留哈希。测试通过不代表语义质量通过。', '',
        'CPU 分支复用真实 SAM3 缓存，GPU 探针确实执行了新的 SAM3 推理。没有新跑 SGA、CLIP 或 LLM；部分缓存沿用早先 SGF 的子类别证据。没有引入 OneFormer。`ENV_REUSE.json` 的 no-model 标志仅指 CPU 复用，GPU 证据单列于 `ENVIRONMENT_SCOPE.json`。', '',
        '## 2. Orbbec 为什么丢失已有实例', '',
        f'从旧 C 到 R3，{loss["lost_points"]:,} 个原本有实例的点失去实例归属。逐点归因如下，五场 R3 重放均与封存输出完全一致：', '',
        '| 最后未通过的阶段 | 丢失点数 |', '|---|---:|']
    reasons = {'all_groups_insufficient_frames': '所有候选组均未达到组帧数', 'frame_eligible_groups_low_score': '有组达到帧数，但组分数低于 0.8',
               'ambiguous_ownership': '归属冲突', 'all_mask_nodes_filtered': '掩码节点全部被过滤', 'owned_semantic_part_below_output_size': '输出片段过小'}
    report += [f'| {label} | {loss["loss_reasons"][key]:,} |' for key, label in reasons.items()]
    report += ['', '前两项占 98.31%。其中 22,424 个“组帧数不足”的点，本身已有至少两个合格输入视角，说明关联没形成合格组是重要环节，不能简单说没有拍到。这里是按每点最后通过阶段作互斥归因，并非放宽某个规则的反事实收益。', '',
        '这些旧实例丢失点均已到达合格投影掩码节点；本轮没有把它们归因于深度失败、前端没有候选或单点视角门槛。但这不代表其他从未识别的设备也不存在前端问题。', '',
        '## 3. 同地图的五场对照', '',
        '下表是实例覆盖率，不是准确率。C 为 `08f8754 / objects_geometry`，R3 为 `f99f359 / consensus_matched`。三个本轮分支所有逐点语义与置信度完全不变。', '',
        '| 场景 | 选帧 | 旧 C | R3 | 三维引导 | 类别分离 | 原始掩码去重＋分离 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        name = row['scene'].split('/')[-1]
        if row['scene'].startswith('orbbec/'): name = 'Orbbec 4812'
        if row['scene'].startswith('3rscan/'): name = '3RScan（仅六帧）'
        report.append(f'| {name} | {row["frames"]} | ' + ' | '.join(pct(row[k]) for k in ('C', 'R3', *STAGES)) + ' |')
    report += ['', '三维引导在 Orbbec 恢复 19,530 个点，每点都有至少两个不同帧的直接掩码支持，R3 已有归属保持不变。恢复类别主要是模型预测的 curtain 9,851 点、curtain-like 4,084 点、floor 4,520 点；这些名称不是人工真值。0030 恢复 2,915 点，其中 wall 占 2,513 点。', '',
        '三维引导使用 C 对象作为暂定三维参照，拒绝重复分离证据和多个显著 R3 锚点；被 R3 判为欠分割的掩码不能提供正支持。歧义帧不参与支持，其他清晰帧仍可支持同一对象。多个 C guide 可能扩展同一个已有 R3 实例，因此“旧 ID 不变”不是“物理对象绝不混合”的保证。', '',
        '## 4. ScanNet0030 的正确性检查', '',
        '保留原评价器字节不变：相同固定对齐、5 cm 几何门槛、99,812 个有效真值点和 27 个对象，排除墙/地面；它是诊断指标，不是官方 AP，0030 也不是盲测场景。', '',
        '| 方案 | GT 实例覆盖 | 实例纯度 | 混合实例数 | 平均最大片段召回 |', '|---|---:|---:|---:|---:|']
    names = {'C': '旧 C', 'R3': 'R3', **STAGES}
    for key, value in legacy.items():
        report.append(f'| {names[key]} | {pct(value["instance_known_coverage"])} | {pct(value["weighted_instance_purity"])} | {value["mixed_instances_secondary_GT_fraction_ge10pct"]} | {pct(value["mean_largest_fragment_recall"])} |')
    report += ['', '三维引导相对 R3：GT 实例覆盖只增加约 0.199 个百分点，纯度约下降 0.149 个百分点，IoU25/50 对象召回不变。因此这只是局部恢复，不能称为显著质量提升。', '',
        '另外统一增加了类别无关诊断：预测实例在其他已知 GT 类别（含墙/地面）的支撑也计入 IoU 并集，采用阈值内最大数量的一对一匹配；相同协议重新评价全部基线。对象类别严格读取推理导出的对象清单，绝不由评价器补名字。', '',
        '| 方案 | 类别无关 IoU50 召回 | 实际导出名称 IoU50 召回 |', '|---|---:|---:|']
    for key in ('C', 'R3', *STAGES):
        value = named[key] if key in named else read(R / f'CLASSAGNOSTIC_0030_{key}.json')
        label = names[key] + ('＋显式命名' if key in named else '')
        report.append(f'| {label} | {round(value["classagnostic_recall_iou50"]*27)}/27 | {round(value["exported_object_semantic_recall_iou50"]*27)}/27 |')
    report += ['', '两个类别分离臂若没有命名步骤，真实导出名称召回都只有 2/27；显式多视角命名后恢复为 11/27、12/27。后者仅多命中一个对象，不能据此认定泛化提升。此前把逐点多数标签用于诊断的结果另存，明确标作事后统计，不计作实际命名能力。', '',
        '显式命名每帧只计一票，至少三帧、类别票占比 ≥0.8；粗类不改成细类。Orbbec raw_unknown 中 40/53 个对象通过命名门槛，但包含 24 个 box、10 个 floor、4 个 wall，专业设备名称仍未解决。点语义地图没有被这些对象级名字重新涂色。', '',
        '0050、0011 的真值标注在本机和 ssh44 已查位置未找到；Orbbec 无标注；因此它们本轮只检查完整性、观测支持和可视化。3RScan 仅六个归档帧，不能声称三类数据集的完整语义准确率全部通过。', '',
        '## 5. 视频跟踪与陌生设备提示', '',
        '- Orbbec 窗帘：共同可见点上的前景时间 IoU 0.8822→0.9568，投影点并集 6,094→6,206。',
        '- ScanNet0030 椅子：合并样/分裂样转变各 8 次→各 0 次，投影并集 8,469→8,472；图中重复片段减少。这些是无 GT 的一致性诊断。',
        '- 3RScan 六帧床：视频丢掉清楚可见的 frame0 床掩码，投影点并集 5,257→2,428。时间 IoU 反而提高，说明不能只优化一致性。稀疏间断输入下的失败不能直接推广到完整连续序列。',
        '- 卷盘文字视频与视觉框可改善部分帧，但后段物体离开视野；非空预测比例不是召回率，正框也未包含全部卷绕线缆。', '',
        '蓝色设备 frame4811 提供了最直接的前端证据：旧 31 条提示没有在其 38,561 个像素上产生有效掩码。相同 SAM3 用 `blue machine` 可分出主体；把这个保存的候选加入旧类别竞争，全部像素仍通过原 0.5 分数/0.1 差值门槛，区域外变化为 0。这个具体漏检来自缺少有效候选，不能靠后端聚类凭空补出来。', '',
        '![蓝色设备提示对照](tracking/figures/blue_equipment_focus.png)', '',
        '蓝色提示和人工框都是按原 RGB 选择的诊断条件，不是自动发现全部类别；像素数量不是准确率，blue machine 也不是设备专业名称。本轮仅验证单帧候选与竞争，没有把新蓝色类别完整投影、跨帧关联并融合回最终地图。', '',
        '## 6. 论文思路与本轮边界', '',
        '| 来源 | 本轮实际验证 | 未声称完成的部分 |', '|---|---|---|',
        '| [Any3DIS · CVPR2025](https://arxiv.org/abs/2411.16183) | 实际 SAM3 图像/视频配对与掩码连续性 | 原文使用 SAM2；未复现完整三维优化与动态规划 |',
        '| [MV3DIS · CVPR2026](https://arxiv.org/abs/2604.08916) | 暂定三维对象引导的保守归属恢复 | 未复现完整超点、掩码选择和连续深度权重 |',
        '| [ConceptGraphs · ICRA2024](https://concept-graphs.github.io/) | 实例与名称分离，显式多视角类别投票 | 未增加 CLIP 外观特征或 LLM 命名 |',
        '| [SAM3](https://arxiv.org/abs/2511.16719) | 描述性文字、视觉正负示例、视频传播 | 无自动业务类别发现，无整场地图采纳 |', '',
        '这些是针对机制的改造与对照，不是论文完整复现。先冻结各臂配置再运行，没有按场景搜索阈值。guided 初次实现遗漏既有欠分割掩码排除，已保留原结果并另跑 `guided_recovery_maskveto`；原臂不参与最终比较。命名步骤为补齐缺失的实际命名而追加，已明确登记为非盲协议扩展。', '',
        '## 7. 速度、版本与下一步取舍', '',
        '新后端是缓存重算：三维引导每场总用时约 0.5–3.5 秒；raw_unknown 包含原深度再投影，约 0.6–22 秒；十次命名总批次 7.16 秒。实际 SAM3 配对探针约 197 秒，蓝色设备补充约 163 秒，均含冷启动与校验。单概念短片段速度不能代替 31 类、整场重建和融合的端到端速度。', '',
        '当前选择：不整体替换 C/R3，不合并到 developnew。保留本轮定位出的修复与实验入口。下一步若继续，优先把描述性候选接入现有地图并用跨帧几何核验；视频作为补充候选，不能无条件删掉图像分支的检测；SGA 应在实例稳定之后再验证关联。', '',
        '本轮代码分支：`experiment/sam3-guided-20260912`，从 `f99f359` 创建；最终提交见 `VERSION.json`。源码与执行脚本进入 Git，数据与效果图保留在本目录。原 C/R3 封存文件校验见 `BASELINE_SEALS_BEFORE.json`、`BASELINE_SEALS_AFTER.json`。', '',
        '详细证据：[损失审计](loss_audit/orbbec/scan_20260909_142829_5ef1fa/backend_loss.json) · [命名](NAMING_SUMMARY.json) · [独立审查](INDEPENDENT_REVIEW.json) · [补充审查](INDEPENDENT_REVIEW_ADDENDUM.json) · [测试](TESTS.json)。', '']
    (R / 'REPORT_zh.md').write_text('\n'.join(report))
    content = markdown.markdown('\n'.join(report), extensions=['tables', 'fenced_code'])
    gallery = ['<h2>实验地图与文件</h2><p>每组图均使用相同原地图、视角和采样；灰色表示未知/未分配。点击展开场景。</p>']
    for row in rows:
        key = row['scene']; gallery.append(f'<details><summary>{html.escape(key)} · {row["frames"]} 个选帧</summary>')
        for stage, label in STAGES.items():
            base = f'{stage}/{key}'
            gallery.append(f'<h3>{label} · 实例覆盖 {pct(row[stage])}</h3><a href="{base}/map/map_labeled.ply">原 RGB＋逐点标签 PLY</a> · <a href="{base}/map/map_instance_id.ply">实例彩色 PLY</a> · <a href="{base}/map/map_semantic_id.ply">语义彩色 PLY</a> · <a href="{base}/objects.json">原始对象清单</a>')
            if stage in named: gallery.append(f' · <a href="{base}/objects_named.json">显式命名清单（独立输出）</a>')
            gallery.append(f'<a href="{base}/map_instance_comparison.png"><img loading="lazy" src="{base}/map_instance_comparison.png" alt="{html.escape(label)} 实例对照"></a>')
        gallery.append('</details>')
    for name, label in [('scannet_chair_000040_comparison.png','椅子：原 RGB / 独立图像 / 视频'), ('3rscan_bed_000000_comparison.png','3RScan 失败例：原 RGB / 独立图像 / 视频')]:
        gallery.append(f'<h3>{label}</h3><img src="tracking/figures/{name}" alt="{label}">')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM3 多方向验证</title><style>body{font-family:system-ui,-apple-system,sans-serif;color:#1d2937;background:#f4f6f8;margin:0;line-height:1.75}main{max-width:1120px;margin:auto;background:white;padding:36px}h1,h2,h3{line-height:1.4;color:#173752}h2{margin-top:42px;border-bottom:1px solid #ddd;padding-bottom:8px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d8dee5;padding:9px;text-align:left}th{background:#edf3f7}img{max-width:100%;height:auto}a{color:#1765ad}details{margin:18px 0;border:1px solid #b8c9da;padding:16px}summary{cursor:pointer;font-weight:650}code{background:#edf2f5;padding:2px 4px}p,li{overflow-wrap:anywhere}@media(max-width:750px){main{padding:16px}table{font-size:11px}td,th{padding:4px}}</style><main>' + content + ''.join(gallery) + '</main></html>'
    (R / 'index.html').write_text(document)
    print(json.dumps({'scenes': len(rows), 'maps': 15, 'quality_accepted': False, 'report': str(R / 'REPORT_zh.md')}))


if __name__ == '__main__':
    main()

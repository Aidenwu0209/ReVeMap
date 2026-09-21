# 集成验证

基线：`2aeacf710c5b9ebf144d0751db9c49a531468570`。本次将2026-09-15冻结几何的有效候选整理成 `refine-semantic`，不改变已有模型注册表、SAM3 stride5、串行/阶段并行调度或SLAM。

- 原语义运行时、命名、融合、多视角及新安全测试：首轮103项通过；随后增加解释器派发/PYTHONPATH测试，新模块8项全部通过（总计104个不同测试）。
- 五场真实封存Qwen/SAM3预测经新代码CPU重放，semantic、instance、confidence及其他存在的数组**逐元素相同**，类别字典也相同。包括0030、0050、0011、Orbbec及3R443。[重放凭证](REPLAY_VERIFICATION.json)
- ssh33通过新 `refine-semantic --stage all --fragments` 完整执行ScanNet0050的选图、238次真实Qwen请求、SAM3确认、未知点写回；退出0，144.23秒，输出semantic/instance/confidence和类别字典与封存候选完全一致。[执行凭证](FRESH_0050.json) / [结果比对](FRESH_0050_VERIFICATION.json)
- 上述144.23秒包括语义阶段读取/加载，不含新SLAM，是已有几何和完整4652帧估计轨迹上的对象语义优化；不能作为完整RGB-D pipeline FPS。
- 新入口要求显式RGB注册契约，并去除了原实验脚本里的固定数据集路径。实际ScanNet映射仍使用同一原sens JPEG及内参公式；原GT pose记录只跳过不解码。
- 对重复帧、不可见点、伪造共识、缺失注册契约拒绝处理；不会用独立验证帧补充投票，不把整实例未观测部分涂满，不改变既有标签和所有权。
- 新模块编译、CLI帮助、`git diff --check`、本次报告相对链接检查通过；实验证据图已检查。未声称浏览器渲染验收。

真实GPU见证时父进程环境已提供绝对src导入路径；随后修正派发器自行计算该绝对路径的层级，并用独立派发测试验证五个子阶段及各解释器。该改动只影响模块查找，不改变任何模型或融合数值。

数据、权重和大型完整PLY保留在本地/ssh33封存目录，不进入Git。实现文件哈希见 [SOURCE_HASHES.json](SOURCE_HASHES.json)。实验原始分数与集成验证分开保存，没有因为封装代码而声称新增独立泛化证据。

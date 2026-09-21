# 运行记录

运行机为用户指定的 ssh44（100.64.57.44），Windows / WSL，RTX 5070 Ti 16GB。沿用已有的 SAM3 GPU 环境和 Open3D CPU 环境；未安装或修改依赖。环境规格文件 SHA256 与已有真实图像推理、独立见证一致，见 ENV_REUSE.json。这里记录的是新实验入口，不是环境重建说明。

远程目录：`/mnt/d/SGF-SGA-experiments/cross_dataset_frame_ablation_20260913_v1`。

已执行的专用 systemd 用户服务：`cross-frame-ablation-20260913-v1.service`。服务按 Orbbec、ScanNet0050、ScanNet0011 顺序运行。

```sh
/usr/bin/python3 -u run_batch.py
```

每场由 run_batch.py 设置 EXPERIMENT_SCENE_DIR 为该场绝对输出目录，GPU 推理与已就绪组的 CPU 重放可以重叠。三个间隔共享逐帧掩码，以排除模型输出变化。

```sh
/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python -u infer_frames.py
/home/aidenwu/Documents/sgf_sga_restore_20260910_v1/sga_env/bin/python -u replay_arm.py --stride 20
/home/aidenwu/Documents/sgf_sga_restore_20260910_v1/sga_env/bin/python -u replay_arm.py --stride 10
/home/aidenwu/Documents/sgf_sga_restore_20260910_v1/sga_env/bin/python -u replay_arm.py --stride 5
```

取帧为0开始的固定帧号间隔，首尾保留，并按时间顺序融合。所有RGB-D、冻结位姿逐帧检查存在。相同旧帧的分割、投影和每20帧最终语义/实例/置信度都必须精确相等。CPU沿用已验证等价的二值交集BLAS计数实现。

本地 collect_loop.py 通过SSH收回已完成的输出，校验传输哈希，导出保持原顶点字段的PLY，计算跨方案变化与固定视角图，更新index.html。没有GT输入，未计算准确率或真实实例召回。新地图不是EVOL-SAM3输出，也没有新做SGA神经网络前向。

原始封存目录不作为输出。复现应使用新的独立目录；已有输出目录不允许覆盖重放。模型时间为所选帧累计成本，完整逐帧运行的冷加载另记MODEL.json，后端耗时单列。

# 用 ORB 半位姿重建第二张实测表面

此命令接收同次 ScanNet 录制的原始 `.sens`、ReVeMap 的原生 DROID 轨迹和原生补全点云 A，输出可作为第二表面的彩色单专家标签图，以及主图顶点逐点对应的半位姿坐标。后者供两张图在重叠区关联实例编号。[Python 入口](../../tools/build_orb_half_surface.py)

## 编译与权重

在 Linux 安装并编译 ORB-SLAM3 所需的 OpenCV、Eigen、Pangolin、Boost serialization、zlib，以及上游附带的 DBoW2、g2o 和 Sophus。实测链接的 ORB-SLAM3 检出版本为 `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`。先按上游说明运行其 `build.sh`，生成共享库并解压 `Vocabulary/ORBvoc.txt`。自定义安装的 Pangolin 或 Boost 通过 CMake 前缀提供，运行时动态库应可解析。[ORB-SLAM3 源码与构建说明](https://github.com/UZ-SLAMLab/ORB_SLAM3)

```bash
cmake -S tools/orb_rgbd -B build/orb_rgbd \
  -DCMAKE_BUILD_TYPE=Release -DORB_SLAM3_ROOT="$ORB_SLAM3_ROOT"
cmake --build build/orb_rgbd -j2
```

几何环境使用 NumPy、SciPy、OpenCV、Open3D、Pillow 和 plyfile。单专家环境另需 CUDA、PyTorch、spconv、segmentator 和 OneFormer3D 的 MMDetection3D 依赖。使用作者发布的 ScanNet20 配置与 v1.0 权重，后者的 SHA-256 为 `9ba46231ff62d831c24d1e42e99e0e8c61978857d7f0475e39e002cce34cb7b6`。词典 SHA-256 为 `f8dd027f7a6cb88129821341194d7f2c75b77b3394257ddd0d2229863d1a3570`。[OneFormer3D 安装说明](https://github.com/filaPro/oneformer3d) · [权重下载与核对记录](checkpoint.json)

## 运行

使用原生 `revemap run-sam3` 的同次输入和估计结果。轨迹必须按照原始传感器帧 0 至末帧提供完整、有效的米制 `T_world_camera_m`。A 点云保留原始 DROID 世界坐标与原顶点顺序。变量 `ONEFORMER_PYTHON` 指向配置好 OneFormer3D 导入路径的解释器。[ReVeMap 输入和输出](../../README.md)

```bash
python tools/build_orb_half_surface.py \
  --sensor "$SENSOR" --trajectory "$ESTIMATED_TRAJECTORY" --native-map "$ORIGINAL_A" \
  --orb-runner build/orb_rgbd/revemap_orb_rgbd --vocabulary "$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt" \
  --expert-python "$ONEFORMER_PYTHON" --config "$ONEFORMER_SCAN20_CONFIG" \
  --checkpoint "$ONEFORMER_SCAN20_CHECKPOINT" --output "$ORB_HALF"
```

| 输出 | 后续用途 |
|---|---|
| `trajectory.json`、`orb_supported.npy` | 半位姿与逐帧 ORB 验收掩码 |
| `mesh.ply`、`A.ply` | 由全部原始深度重融的三角网格和多视角补全表面 |
| `single/map_labeled.ply`、`single/panoptic.npz` | 单专家在新 A 点序上的类别与实例，可将 PLY 传作合成器的 `--secondary` |
| `primary_in_secondary.ply` | 与 `--native-map` 同点序的重叠区关联坐标，可作合成器的 `--primary-in-secondary` |
| `geometry.json`、`single/result.json`、`result.json` | 输入、几何、权重和标签输出的哈希及帧数 |

ORB 前端把原始彩色画面按传感器内参注册至深度成像平面，按帧号使用 30 Hz 名义时钟。两套后端导出均保留且跟踪状态为 OK 的帧才参与修正。将最大连通图与 TUM 轨迹置于共同相机系后，两者须在 2 cm 和 0.5° 内吻合。首个通过验收的相机将 ORB 对齐到 DROID 的估计世界系。相机中心取中点，朝向取最短旋转的中点。其余帧沿用 DROID。TSDF 用 2 cm 体素、8 cm 截断距离和 4.5 m 深度上限重融，再用具有视角分离的多帧实测深度补点。单专家以全部相机向上方向中位数初始化，通过 30°、18°、12° 的面法向筛选确定竖直方向。

推理会跳过 `.sens` 每帧保存的参考位姿。评分阶段才使用存档约定的首对有效参考相机，按 `T_ref[k] @ inv(T_estimated[k])` 对齐；生产输出保持 DROID 世界系。非零退出时保留首次 ORB 日志并以相同设置再运行一次。重复独立运行时，ORB 的异步建图和闭环会产生轨迹差异。使用 `--orb-replay DIR` 代替 `--orb-runner` 与 `--vocabulary` 可重放特定 ORB 会话，其目录需要三份导出和 `ORB_RAW_SEALED.json`。输入流及导出均校验 SHA-256。`--geometry-only` 可先生成几何，再用同一输出目录补跑单专家。[实测对应](orb-half-scene0062.json)

实测 scene0062_00 的 730 帧全部参与重融。复用原 ORB 会话时，85,375 个重融点的坐标和颜色、全部 94,618 个主图对应坐标、730 个位姿、三角网格字节，以及 85,375 个单专家类别和实例与历史输出完全一致，PQ 为 61.968%。从原始流重新启动 ORB 时得到 85,064 个重融点，首对相机一致，独立运行 PQ 为 62.631%。

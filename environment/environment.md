# 训练与评价环境快照

**这是环境快照，不是可保证重建的 lockfile。** 下表记录的是当时实际运行环境的关键版本，
用于解释数值来源与排查兼容性问题；它不包含逐包哈希、不锁定传递依赖，也不保证在其他机器上
逐位复现。请勿把它当作可复现构建的依赖清单。

## 1. 信息来源

- `项目离线保全_20261105/code_snapshot/grsl_work_20261103/release_scale2_v2_20261104/env/environment.txt`
  （Python / torch / timm / numpy / h5py / scikit-learn 版本，GPU 与驱动信息，以及当时的 `pip freeze`）
- `项目离线保全_20261105/code_snapshot/grsl_work_20261103/q1_norm_stage1_v1/environment.txt`
  （N2 阶段记录：Python 3.12.3、PyTorch 2.7.0+cu128、CUDA 12.8、RTX 5090、timm 1.0.27、torchvision 0.22.0+cu128）

两份记录来自同一类 GPU 实例，关键版本一致；本文件只归纳二者共同的、与训练/评价直接相关的项，
没有拼接出一个虚构的统一锁文件。

## 2. 关键版本

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.3 | 两个阶段一致 |
| PyTorch | 2.7.0+cu128 | CUDA 12.8 构建 |
| torchvision | 0.22.0+cu128 | 与 torch 配套 |
| timm | 1.0.27 | Swin-T 骨干与 ImageNet 起点 |
| NumPy | 2.4.6 | 数据归一化与指标计算 |
| h5py | 3.16.0 | 读取 Landslide4Sense 的 HDF5 样本 |
| safetensors | 0.8.0 | 读取 ImageNet 起点的 `.safetensors` 权重 |
| scikit-learn | 1.9.1 | 仅用于环境自检，训练与评价不依赖 |
| CUDA | 12.8 | 由 cu128 轮子与驱动共同决定 |
| GPU | NVIDIA GeForce RTX 5090（32,607 MiB） | 计算能力 sm_120，需要 cu128 及以上轮子 |
| 显卡驱动 | 595.71.05 | 记录值，非硬性下限 |
| 操作系统 | Linux 5.15 系列（glibc 2.35） | 仅记录，不构成要求 |

训练入口在启动时会把 `torch` 版本写入 `config.json`；评价入口会把 `torch`、`timm`、`numpy`、
`h5py` 的实际版本写入 `protocol.json`（`reproducibility.evaluate` 还记录 `torchvision`），
可用于回查每次运行的真实环境。

## 3. 未记录或无法保证的部分

- 未记录完整的 `pip freeze` 逐包哈希，也未记录 conda 环境导出文件；
- 未记录 CUDA 工具链、cuDNN 的具体小版本；
- 未记录 CPU 型号、内存与磁盘；
- 不同批次运行可能落在不同实例上，GPU 型号相同但驱动版本可能不同。

因此，本快照只能支持"按上述主版本重建一个等价环境"，**不能**支持"逐位一致的确定性重建"。
论文中的数值结论以已保存的 checkpoint 与逐样本评价记录为准，不以环境完全一致为前提。

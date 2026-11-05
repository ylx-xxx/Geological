# 结果数据说明

本目录随论文发布提供**已完成的冻结实验**的逐样本评价结果与配对 patch 重抽样汇总。所有数值均为原始分析产物，本次发布**没有重新运行 bootstrap、没有训练、没有推理**。

## 1. 目录与命名

```
results/
├── inputs/<split>__<config>__<seed>/
│   ├── per_sample.csv
│   ├── metrics.json
│   └── protocol.json
├── bootstrap_summary.csv
├── per_seed_effects.csv
└── README.md
```

- `<split>`：`valid` 或 `test`。
- `<config>`：训练/推理时使用的配置代号，与论文配置名的对应关系见下表。
- `<seed>`：`2026`、`2027`、`2028`。

| 目录中的配置代号 | 论文配置名 | 输入通道 | 输入尺度 | 地形归一化 |
|---|---|---|---|---|
| `RGB-S1` | RGB-S1-N0 | RGB | 1× | 否 |
| `F14-S1` | full14-S1-N0 | 14 通道 | 1× | 否 |
| `RGB-S2` | RGB-S2-N0 | RGB | 2× | 否 |
| `F14-S2` | full14-S2-N0 | 14 通道 | 2× | 否 |
| `F14-S2-N2` | full14-S2-N2 | 14 通道 | 2× | 是（N2） |

数据集规模：Train 3799、Valid 245、历史 Test 800（单位均为 patch）。模型检查点**只用 Valid 选模**；三视角 TTA。

## 2. 单个 run 的文件

### `per_sample.csv`

每个 patch 一行，共 245 行（Valid）或 800 行（Test）。主要列：

- `sample`：patch 文件名。
- `positive_pixels` / `valid_pixels`：该 patch 内滑坡正类像素数与有效像素数。
- `TP`、`FP`、`FN`、`TN`：该 patch 的混淆计数。
- `Landslide_IoU`、`Landslide_F1`、`Landslide_Precision`、`Landslide_Recall`：**逐 patch** 的滑坡类指标。
- `mIoU`、`OA`：逐 patch 的全体类别平均 IoU 与总体精度（含背景类）。
- `foreground_patch_iou`：该 patch 的前景 IoU。

注意：逐 patch 的 `Landslide_IoU` 与整个 split 的 pooled IoU **不是同一个量**。本发布中的配置级 IoU（`metrics.json`、`paper_main_table.csv`、bootstrap）一律使用 pooled 定义：

```
Landslide IoU = ΣTP / (ΣTP + ΣFP + ΣFN)
```

即先在 patch 上求和，再相除；不是逐 patch IoU 的算术平均。

### `metrics.json`

该 `<split>__<config>__<seed>` run 在**整个 split 上 pooled** 的评价结果：`Landslide_IoU`、`Landslide_F1`、`Landslide_Precision`、`Landslide_Recall`、`mIoU`、`OA`、混淆矩阵计数、样本数与推理耗时（`seconds`）。`seconds` 依赖硬件与运行环境，不属于科学比较指标。

### `protocol.json`

该 run 的推理协议：检查点 SHA256、split、输入尺度、通道、归一化方式、TTA 视角数（3）、AMP、检查点加载方式、样本数、软件版本，以及推理源码的 SHA256。N2 配置另有地形统计文件及其 SHA256。

## 3. 汇总表

### `bootstrap_summary.csv`

四个固定对比的配对 patch bootstrap 结果，每个对比一行：

| 对比 ID | 对比 | 角色 |
|---|---|---|
| C1 | full14-S2 − RGB-S2 | 主对比：多通道效应 |
| C2 | full14-S2-N2 − full14-S2 | 次对比：地形归一化效应 |
| R1 | RGB-S2 − RGB-S1 | 参照：RGB 尺度效应 |
| R2 | full14-S2 − full14-S1 | 参照：full14 尺度效应 |

主要列：

- `delta_valid_point` / `delta_test_point`：三个种子平均后的 ΔValid、ΔTest 点估计（IoU 原值，范围 0–1）。
- `delta_valid_lo` / `delta_valid_hi`、`delta_test_lo` / `delta_test_hi`：95% 百分位区间（2.5% / 97.5%）。
- `D_point = delta_test_point − delta_valid_point`，`D_lo` / `D_hi` 为其区间。
- `n_reps = 5000`：bootstrap 重复次数。
- `valid_reps_valid_split`、`valid_reps_test_split`、`invalid_reps`：有效/无效重复数；本次无效重复为 0。
- `boot_mean_*`、`sd_*_reps`：bootstrap 分布的均值与标准差。
- `seeds_delta_valid_positive`、`seeds_delta_test_positive`、`seeds_D_positive`：3 个种子中对应差值大于 0 的个数（0–3），用于展示种子间方向一致性。

**bootstrap 定义**（详见 `../provenance/input_manifest.json`）：以 patch 为重抽样单位、有放回抽样；同一 split、同一重复内所有配置与种子共用同一组 patch 索引；每次重复先对 TP/FP/FN 求和得到 pooled IoU，再做同种子差值，最后对 3 个种子取平均；区间取重复分布的第 2.5/97.5 百分位。

### `per_seed_effects.csv`

四个对比 × 三个种子的逐种子结果：左右配置的 Valid/Test pooled IoU（`iou_left_*`、`iou_right_*`）、逐种子差值 `delta_valid` / `delta_test`、`D`，以及该对比逐种子 ΔValid 的样本标准差 `delta_valid_seed_sd`。它用于检查效应方向是否由某单一模型种子驱动。

## 4. 这些数字怎么读（重要限制）

1. **条件于固定检查点。** bootstrap 只对评价 patch 重抽样，不包含训练随机性、检查点选择或超参数选择的不确定性。它回答的是“在这三个已冻结的模型上，若把 Valid/Test patch 视为可交换样本，性能差有多稳定”，不是“换一个训练种子效应会怎样”。
2. **可交换性假设。** 区间把 patch 当作独立可交换单位，没有做空间自相关、事件级或影像级聚类修正。滑坡 patch 之间通常存在空间相关性，因此这些区间很可能**偏窄**，应视为**探索性区间**，而不是保守的不确定性下界。
3. **不是显著性检验。** 文中未计算 p 值，也不应把“区间跨零/不跨零”读成“无效应/有效应”的证明。区间跨零只表示在当前假设与样本下方向不确定；不跨零也不等于排除了偏倚或确认了因果效应。点估计始终照实报告，未按“是否显著”筛选或改色。
4. **历史 Test 不是全新独立确认。** Test 为 800 个 patch 的历史基准，在研究过程中已参与过研究反馈与方案迭代；虽然选模只用 Valid、Test 未参与选模，但 Test 仍不是未触碰过的独立确认集。Test 上的区间不能恢复独立性，也不能修正自适应复用带来的偏倚。
5. **只有 3 个训练种子。** `paper_main_table.csv` 中的种子标准差只是 3 个种子的描述性离散度，不是置信区间；不要把它与 bootstrap 区间混用。
6. **单位。** CSV 中为 IoU 原值（0–1）；图 G2 中乘以 100，单位为“IoU 百分点”。

## 5. 完整性与脱敏

- `results/inputs/` 共 30 组 × 3 个文件 = 90 个数据文件。
- 其中 84 个与源归档**逐字节一致**（30 个 `per_sample.csv`、30 个 `metrics.json`、24 个 `protocol.json`），其 SHA256 与 `../provenance/input_manifest.json` 记录的值一致。
- 6 个 N2 配置的 `protocol.json`（`valid|test` × 3 seeds）仅将 `terrain_stats_file` 中的服务器绝对路径前缀脱敏为 `<REMOTE_DATA_ROOT>`，其余字节与源文件一致；因此这 6 个文件的计算 SHA256 会与 `input_manifest.json` 中记录的原始 SHA256 不同。原始哈希未被修改，仍保留在清单中，供核对源归档。
- `bootstrap_summary.csv`、`per_seed_effects.csv` 与源归档逐字节一致。
- 本目录不含原始影像、H5、GT、预测数组或模型权重。

逐文件来源与 SHA256 见 `../provenance/release_source_map_results.csv`。

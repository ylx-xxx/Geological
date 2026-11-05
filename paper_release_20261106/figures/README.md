# 图件与绘图脚本

本目录包含论文图 G1–G3 的 PDF/PNG、论文表格、绘图脚本及脚本直接依赖的历史训练曲线 CSV。所有图只描述已完成的冻结实验；未改阈值、未重选模型、未补训练。

## 文件

```
figures/
├── G1_absolute_iou.pdf / .png
├── G2_paired_effects.pdf / .png
├── G3_training_curves.pdf / .png
├── paper_main_table.csv
├── paper_effects_table.csv
├── make_figures.py
├── source_data/history/<config>__<seed>.csv   （15 份）
└── README.md
```

> 源归档中另有同名 `.svg` 矢量图；本次按发布要求只收录 PDF 与 PNG。运行 `make_figures.py` 会同时重新生成 `.pdf`、`.svg`、`.png`。

## 每张图对应的数据

| 图 | 内容 | 直接数据源（发布目录内相对路径） |
|---|---|---|
| G1 `G1_absolute_iou` | 五配置在 Valid（245 patch）与历史 Test（800 patch）上的 pooled Landslide IoU；散点为单个训练种子，误差棒为 3 个种子的样本标准差 | `../results/inputs/valid__<config>__<seed>/per_sample.csv`、`../results/inputs/test__<config>__<seed>/per_sample.csv`（共 30 份）；脚本按 `ΣTP/(ΣTP+ΣFP+ΣFN)` 重算 pooled IoU |
| G2 `G2_paired_effects` | 四对比的 ΔValid、ΔTest、D 点估计与 5000 次配对 patch bootstrap 的 95% 百分位区间（IoU 百分点） | `../results/bootstrap_summary.csv` |
| G3 `G3_training_curves` | 三个种子分面的 60 轮 Valid Landslide IoU 原始曲线；星号为实际用于选模的 best-Valid epoch | `source_data/history/<config>__<seed>.csv`（15 份） |
| 主表 `paper_main_table.csv` | 五配置逐种子 Valid/Test pooled IoU、均值与样本 SD | 同 G1 的 30 份 `per_sample.csv` |
| 效应表 `paper_effects_table.csv` | 四对比的点估计、区间、有效重复数与逐种子符号计数 | `../results/bootstrap_summary.csv` |

`source_data/history/` 中 `<config>` 取 `RGB-S1`、`F14-S1`、`RGB-S2`、`F14-S2`、`F14-S2-N2`，`<seed>` 取 `2026`、`2027`、`2028`。这 15 份文件来自各配置实际训练目录的 `history.csv`，原训练目录的归档相对路径为：

| config | 原训练目录（相对快照根） |
|---|---|
| RGB-S1 | `grsl_work_20261103/lr_screening_20261103/runs/loveda_encoder_lr1e-4_s<seed>` |
| F14-S1 | `grsl_work_20261103/mc_v1/runs/F14-L-<seed>` |
| RGB-S2 | `grsl_work_20261103/channel_scale_2x2_20261104/runs/RGB-L-S2-<seed>` |
| F14-S2 | `grsl_work_20261103/scale2_v2/runs/F14-L-S2-<seed>` |
| F14-S2-N2 | `grsl_work_20261103/q1_norm_stage1_v1/runs/F14-L-S2-N2-<seed>` |

## 如何重画

依赖：Python 3、NumPy、Matplotlib。以下命令均在 `paper_release_20261106/figures/` 目录下执行（相对路径）。

`make_figures.py` 的 G3 部分按“快照根/训练目录/history.csv”读取文件，因此需先把随附的 15 份 CSV 按上表映射摆成快照目录树：

```bash
cd paper_release_20261106/figures
SNAP=snapshot/grsl_work_20261103

stage() {  # 参数：config seed 相对目录
  d="$SNAP/$3"
  mkdir -p "$d"
  cp "source_data/history/$1__$2.csv" "$d/history.csv"
}

for seed in 2026 2027 2028; do
  stage RGB-S1     "$seed" "lr_screening_20261103/runs/loveda_encoder_lr1e-4_s$seed"
  stage F14-S1     "$seed" "mc_v1/runs/F14-L-$seed"
  stage RGB-S2     "$seed" "channel_scale_2x2_20261104/runs/RGB-L-S2-$seed"
  stage F14-S2     "$seed" "scale2_v2/runs/F14-L-S2-$seed"
  stage F14-S2-N2  "$seed" "q1_norm_stage1_v1/runs/F14-L-S2-N2-$seed"
done

python -I -X utf8 make_figures.py --tables ../results --snapshot ./snapshot --out .
```

说明：

- `--tables ../results` 提供 G1 所需的 `inputs/` 与 G2 所需的 `bootstrap_summary.csv`。
- `--out .` 会把三张图的 PDF/SVG/PNG 以及两张论文表格写回当前目录，覆盖同名文件。若只想检查，可把 `--out` 指向临时目录，例如 `--out /tmp/fig_check`。
- 脚本不重新推理、不重跑 bootstrap，只从既有逐样本表和汇总表读取数值；G1 的 pooled IoU 是按 `TP/(TP+FP+FN)` 从 `per_sample.csv` 重算的。
- G2 脚注中的“区间跨零不等于无效应、固定检查点、patch 可交换、Test 为历史基准”等限制与 `../results/README.md` 一致。
- 脚本中不含本机或服务器绝对路径。

逐文件来源与 SHA256 见 `../provenance/release_source_map_results.csv`。

# 分析源码（随附，本次未重跑）

本目录保存生成 `../results/bootstrap_summary.csv` 与 `../results/per_seed_effects.csv` 的**已执行源码**，按原归档的相对路径 `code/` 组织。**本次发布只随附源码，没有重新运行 bootstrap、诊断或测试**；所有结果均为原分析产物，数值未改动。

## 文件

| 文件 | 作用 |
|---|---|
| `code/bootstrap_analysis.py` | 主分析。读取 30 份 `per_sample.csv`，计算全量 pooled IoU，执行 B=5000 的配对 patch bootstrap，输出 `bootstrap_summary.csv`、`per_seed_effects.csv` 与 `input_manifest.json`。 |
| `code/diagnostics.py` | 补充诊断。用与主分析完全相同的随机流复算 bootstrap，输出均值一致性、分位数稳定性、符号比例等诊断量（原归档中对应 `results/diagnostics.json`）。 |
| `code/test_analysis.py` | 必需测试：表格乱序后的身份配对、pooled IoU 与 patch 平均不可混用、同模型差为零、点估计可复算。原分析运行记录为 4 项全部通过。 |

## 原分析的关键设定

- 重抽样单位：patch（有放回）；B=5000；基础随机种子 20261105。
- Valid 与 Test 使用独立随机流：`SeedSequence([seed, 0])` 与 `SeedSequence([seed, 1])`。
- 同一 split、同一 replicate 内，所有配置与种子共用同一组 patch 索引。
- 统计量：先对重抽样后的 TP/FP/FN 求和得到 pooled IoU，再做同种子配置差，最后对 3 个种子取平均。
- 区间：重复分布的 2.5% / 97.5% 百分位（`numpy.percentile`，线性插值）。
- 脚本只读取已导出的逐样本评价表，不做模型推理。

## 如仅作只读复核（可选，本次未执行）

原归档中的调用方式等价于：

```bash
# 主分析（工作目录为 analysis/）
python -I code/bootstrap_analysis.py --input-root ../results/inputs --out-dir ../results

# 补充诊断
python -I code/diagnostics.py ../results/inputs ../results

# 必需测试
python -I code/test_analysis.py --input-root ../results/inputs
```

说明：

- 上述命令**本次未运行**，仅记录源码原本的调用方式。
- 若执行主分析，会覆盖 `../results` 下的汇总表并重写 `input_manifest.json`，同时因发布目录中没有 `inputs/_manifests/` 而使 `source_manifests` 为空。除非确需复核，否则不要覆盖发布结果；发布数值以现有文件为准。
- `test_analysis.py` 会把输入树复制到临时目录后做乱序/恒等检查，不修改 `results/inputs/`。
- 三个脚本中不含本机或服务器绝对路径。

## 依赖

NumPy（原分析环境为 Python 3.12.7、NumPy 1.26.4）。未使用 GPU，未调用模型代码。

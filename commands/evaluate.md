# 评价命令

评价入口按实际执行记录选择：

| 配置 | 评价代码版本 | 入口 | `--input-scale` |
|---|---|---|---|
| RGB-S1-N0、full14-S1-N0 | `code_versions/s2_n0`（即归档 `code_scale2_v2/`） | `reproducibility.eval_scale2` | 1 |
| RGB-S2-N0、full14-S2-N0 | `code_versions/s2_n0` | `reproducibility.eval_scale2` | 2 |
| full14-S2-N2 | `code_versions/s2_n2`（即归档 `code_q1_norm_v1/`） | `reproducibility.eval_scale2` | 2 |

说明：冻结归档中 12 个 N0 运行的已登记 Valid 评价，统一由 `code_scale2_v2/reproducibility/eval_scale2.py`
完成（`--input-scale 1` 即原生 128 路径，`--input-scale 2` 为尺度包装器路径），各运行的
`protocol.json` 中记录了该评价脚本的 SHA256，可用于核对。因此本文把 `eval_scale2` 作为主入口。

共同约定：

```bash
export L4S_DATA_ROOT=/path/to/landslide4Sense
export CKPT_ROOT=/path/to/checkpoints          # 各运行的 best.pt
export EVAL_ROOT=/path/to/evals                # 评价输出根目录
export TERRAIN_STATS=/path/to/terrain_n2_stats.npz  # 仅 N2 需要
```

- `--expected-sha256` 是冻结身份校验：程序先对 checkpoint 计算 SHA256，与传入值不一致就报错退出。
  15 个 target checkpoint 的 SHA256 见仓库根目录 `weights_index.csv`。
- `--out` 必须是**不存在**的新目录；已有评价结果不可覆盖。
- `--tta 1`（默认）表示三视角 TTA：原图、水平翻转、垂直翻转，翻转回并平均 logits；
  `--tta 0` 为单视角。正式结果均使用三视角。
- `--split val` 用于 Valid 复评；`--split test` 必须先加 `--confirm-test-once`。
- 选模已在训练阶段按 Valid 前景 IoU 完成；评价入口**不重新选模、不搜索阈值**。
- 所有命令都以模块方式运行：先 `cd` 到对应代码版本目录，再 `python -m reproducibility.eval_scale2`。

## 1. RGB-S1-N0 与 full14-S1-N0（`--input-scale 1`）

```bash
# RGB-S1-N0，seed 2026
cd code_versions/s2_n0
python -m reproducibility.eval_scale2 \
  --data-root "$L4S_DATA_ROOT" \
  --checkpoint "$CKPT_ROOT/RGB-S1-N0/2026/best.pt" \
  --expected-sha256 <weights_index.csv 中该运行的 checkpoint_sha256> \
  --out "$EVAL_ROOT/RGB-S1-N0/2026" \
  --split val \
  --input-scale 1 \
  --channels rgb \
  --batch-size 64 \
  --workers 4 \
  --tta 1 \
  --amp 1
```

```bash
# full14-S1-N0，seed 2026
cd code_versions/s2_n0
python -m reproducibility.eval_scale2 \
  --data-root "$L4S_DATA_ROOT" \
  --checkpoint "$CKPT_ROOT/full14-S1-N0/2026/best.pt" \
  --expected-sha256 <weights_index.csv 中该运行的 checkpoint_sha256> \
  --out "$EVAL_ROOT/full14-S1-N0/2026" \
  --split val \
  --input-scale 1 \
  --channels full14 \
  --batch-size 64 \
  --workers 4 \
  --tta 1 \
  --amp 1
```

seed 2027、2028 只改 `--checkpoint` 与 `--out` 中的 seed。

## 2. RGB-S2-N0 与 full14-S2-N0（`--input-scale 2`）

```bash
# RGB-S2-N0，seed 2026
cd code_versions/s2_n0
python -m reproducibility.eval_scale2 \
  --data-root "$L4S_DATA_ROOT" \
  --checkpoint "$CKPT_ROOT/RGB-S2-N0/2026/best.pt" \
  --expected-sha256 <weights_index.csv 中该运行的 checkpoint_sha256> \
  --out "$EVAL_ROOT/RGB-S2-N0/2026" \
  --split val \
  --input-scale 2 \
  --channels rgb \
  --batch-size 64 \
  --workers 4 \
  --tta 1 \
  --amp 1
```

```bash
# full14-S2-N0，seed 2026
cd code_versions/s2_n0
python -m reproducibility.eval_scale2 \
  --data-root "$L4S_DATA_ROOT" \
  --checkpoint "$CKPT_ROOT/full14-S2-N0/2026/best.pt" \
  --expected-sha256 <weights_index.csv 中该运行的 checkpoint_sha256> \
  --out "$EVAL_ROOT/full14-S2-N0/2026" \
  --split val \
  --input-scale 2 \
  --channels full14 \
  --batch-size 64 \
  --workers 4 \
  --tta 1 \
  --amp 1
```

## 3. full14-S2-N2（`--input-scale 2`，`--norm-mode n2`）

N2 只改变地形两个通道的归一化，评价时必须传入与训练相同的 `--norm-mode n2` 与 `--terrain-stats`，
否则数值口径与训练不一致。

```bash
cd code_versions/s2_n2
python -m reproducibility.eval_scale2 \
  --data-root "$L4S_DATA_ROOT" \
  --checkpoint "$CKPT_ROOT/full14-S2-N2/2026/best.pt" \
  --expected-sha256 <weights_index.csv 中该运行的 checkpoint_sha256> \
  --out "$EVAL_ROOT/full14-S2-N2/2026" \
  --split val \
  --input-scale 2 \
  --channels full14 \
  --batch-size 64 \
  --workers 4 \
  --tta 1 \
  --amp 1 \
  --norm-mode n2 \
  --terrain-stats "$TERRAIN_STATS"
```

## 4. 历史 Test（800 张）

把 `--split val` 改为 `--split test --confirm-test-once`，其余参数与对应 Valid 评价完全相同。
Test 评价只能在冻结 checkpoint 身份之后执行一次；本发布不重新运行任何 Test 评价。

## 5. 另一个评价入口（仅 S1，等价但字段较少）

`code_versions/rgb_s1` 与 `code_versions/full14_s1` 各自保留了 `reproducibility.evaluate`
（即归档 `code/reproducibility/evaluate.py`）。它与 `eval_scale2` 共用同一套
`EvaluationDataset`、`predict`（三视角 TTA）与混淆计数函数；两个版本的网络类定义相同，
差异只在 patch 核迁移函数的签名，因此加载同一 checkpoint 时前向与指标口径一致
（本次发布未重新运行该入口做数值比对）。区别是不写 `input_scale` 字段、不计算前景概率 AP，
`per_sample.csv` 多记录图像/掩膜 SHA256。用法示例：

```bash
cd code_versions/rgb_s1
python -m reproducibility.evaluate \
  --data-root "$L4S_DATA_ROOT" \
  --checkpoint "$CKPT_ROOT/RGB-S1-N0/2026/best.pt" \
  --expected-sha256 <weights_index.csv 中该运行的 checkpoint_sha256> \
  --out "$EVAL_ROOT/RGB-S1-N0/2026-evaluate" \
  --split val \
  --channels rgb \
  --batch-size 64 \
  --workers 4 \
  --tta 1 \
  --amp 1 \
  --channels-last 1
```

已登记结果的 `protocol.json` 来自 `eval_scale2`；若需要与登记结果逐字段对照，请使用第 1 节的命令。

## 6. 输出内容

两种入口都会写出 `per_sample.csv`、`metrics.json` 与 `protocol.json`：

- `per_sample.csv`：逐样本混淆计数与指标，以及前景/有效像素数；
  其中 `reproducibility.evaluate` 额外记录图像与掩膜的 SHA256，
  `reproducibility.eval_scale2` 不记录这两个哈希；
- `metrics.json`：全局指标与混淆矩阵，并断言逐样本计数之和等于全局计数；
  `eval_scale2` 另外记录前景概率的平均精度（`Landslide_AP`）；
- `protocol.json`：checkpoint SHA256、split、通道/归一化标记、TTA 视角数、
  `input_scale`（`eval_scale2`），以及评价所用源文件的 SHA256，用于事后审计。

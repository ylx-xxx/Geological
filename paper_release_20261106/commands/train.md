# 训练命令

本文件给出五个正式配置的命令模板。所有数据、权重、输出目录都通过环境变量或显式参数传入，
不依赖任何服务器或个人绝对路径。

## 0. 前置约定

```bash
export L4S_DATA_ROOT=/path/to/landslide4Sense   # 含 TrainData/ValidData/TestData 的数据根目录
export LOVEDA_CKPT=/path/to/loveda_best.pth     # LoveDA 源域编码器权重
export RUN_ROOT=/path/to/runs                   # 训练输出根目录
export TERRAIN_STATS=/path/to/terrain_n2_stats.npz  # 仅 N2 需要：锁定 Train 统计的地形归一化参数
```

- 每个版本目录都保留原有相对路径，因此**必须在对应版本目录内以模块方式运行**：
  `cd code_versions/<版本>` 后执行 `python -m reproducibility.train`。
  直接 `python reproducibility/train.py` 会因相对导入失败（历史运行记录中曾出现同样问题）。
- `--out` 必须是不存在的新目录；程序会拒绝写入已有目录或含 `checkpoints` 的路径。
- 三个 seed 分别是 2026、2027、2028；除 `--seed` 与 `--out` 外，同配置的其余参数完全一致。
- 训练固定 60 轮，按 Valid 前景 IoU 选 best；三视角 TTA 只用于评价，不参与选模。
- 以下命令与 `configs/` 中对应 `config.json` 的 `argv` 一致（仅把绝对路径替换为占位）。
- `--adapter none` 与 `--empty-penalty-weight 0.0` 表示 STCA 与额外前景惩罚**未启用**；
  full14 系列使用 `--patch-extra zero`，即额外通道零初始化（均值初始化未启用）。
  RGB 序列保留历史字段 `--patch-extra mean`，但 RGB 输入与源权重同为 3 通道，
  该字段不会产生任何额外通道。

## 1. RGB-S1-N0（物理 RGB，原生 128）

代码版本：`code_versions/rgb_s1`（对应归档 `code/`）。

```bash
cd code_versions/rgb_s1
python -m reproducibility.train \
  --data-root "$L4S_DATA_ROOT" \
  --out "$RUN_ROOT/RGB-S1-N0/2026" \
  --channels rgb \
  --initialization loveda_encoder \
  --patch-extra mean \
  --protocol train_only \
  --seed 2026 \
  --epochs 60 \
  --max-seconds 3600 \
  --lr 0.0001 \
  --min-lr 1e-6 \
  --loveda-checkpoint "$LOVEDA_CKPT"
```

说明：`--batch-size` 与 `--workers` 未显式给出，取入口默认值 64 与 4（与真实运行一致）。
`--max-seconds 3600` 是该组运行的真实设定。

## 2. full14-S1-N0（全部 14 通道，原生 128）

代码版本：`code_versions/full14_s1`（对应归档 `code_multichannel_v1/`）。

```bash
cd code_versions/full14_s1
python -m reproducibility.train \
  --data-root "$L4S_DATA_ROOT" \
  --out "$RUN_ROOT/full14-S1-N0/2026" \
  --channels full14 \
  --patch-extra zero \
  --patch-channel-map rgb \
  --initialization loveda_encoder \
  --loveda-checkpoint "$LOVEDA_CKPT" \
  --lr 1e-4 \
  --epochs 60 \
  --batch-size 64 \
  --seed 2026 \
  --max-seconds 0 \
  --empty-penalty-weight 0.0 \
  --protocol train_only \
  --workers 4
```

## 3. RGB-S2-N0（物理 RGB，模型内部 128→256→128）

代码版本：`code_versions/s2_n0`（对应归档 `code_scale2_v2/`）。

```bash
cd code_versions/s2_n0
python -m reproducibility.train \
  --data-root "$L4S_DATA_ROOT" \
  --out "$RUN_ROOT/RGB-S2-N0/2026" \
  --channels rgb \
  --patch-channel-map first3 \
  --patch-extra mean \
  --initialization loveda_encoder \
  --loveda-checkpoint "$LOVEDA_CKPT" \
  --lr 1e-4 \
  --min-lr 1e-6 \
  --epochs 60 \
  --batch-size 64 \
  --seed 2026 \
  --max-seconds 0 \
  --empty-penalty-weight 0.0 \
  --adapter none \
  --protocol train_only \
  --workers 4 \
  --input-scale 2
```

说明：`--input-scale 2` 只是模型内部的尺度包装器（128 上采样到 256 推理，再把 256 的 logits
折回 128），数据与标签仍是 128×128，不是新的观测分辨率。RGB 输入只有 3 个通道，
`first3` 表示不做通道映射（程序只在多通道输入且显式指定 `rgb` 时才做物理通道映射）；
通道顺序由 `--channels rgb` 决定，即 [3,2,1]。

## 4. full14-S2-N0（全部 14 通道，模型内部 128→256→128）

代码版本：`code_versions/s2_n0`（对应归档 `code_scale2_v2/`）。

```bash
cd code_versions/s2_n0
python -m reproducibility.train \
  --data-root "$L4S_DATA_ROOT" \
  --out "$RUN_ROOT/full14-S2-N0/2026" \
  --channels full14 \
  --patch-extra zero \
  --patch-channel-map rgb \
  --initialization loveda_encoder \
  --loveda-checkpoint "$LOVEDA_CKPT" \
  --lr 1e-4 \
  --epochs 60 \
  --batch-size 64 \
  --seed 2026 \
  --max-seconds 0 \
  --empty-penalty-weight 0.0 \
  --protocol train_only \
  --workers 4 \
  --input-scale 2
```

## 5. full14-S2-N2（全部 14 通道，S2 尺度，地形通道固定 Train 统计）

代码版本：`code_versions/s2_n2`（对应归档 `code_q1_norm_v1/`）。

```bash
cd code_versions/s2_n2
python -m reproducibility.train \
  --data-root "$L4S_DATA_ROOT" \
  --out "$RUN_ROOT/full14-S2-N2/2026" \
  --channels full14 \
  --patch-extra zero \
  --patch-channel-map rgb \
  --initialization loveda_encoder \
  --loveda-checkpoint "$LOVEDA_CKPT" \
  --lr 1e-4 \
  --epochs 60 \
  --batch-size 64 \
  --seed 2026 \
  --max-seconds 0 \
  --empty-penalty-weight 0.0 \
  --protocol train_only \
  --workers 4 \
  --input-scale 2 \
  --norm-mode n2 \
  --terrain-stats "$TERRAIN_STATS"
```

说明：N2 只把坡度与高程两个通道（12、13）改为锁定 Train 统计的归一化；
其余 12 个光谱通道的裁剪与标准化与 N0 完全相同。`--terrain-stats` 指向的 `.npz`
必须与 N2 训练时使用的地形统计一致（其 SHA256 见对应 `config.json` 的
`terrain_stats_sha256` 字段）。

## 6. 逐 seed 运行

把上面的 `--out` 改为 `"$RUN_ROOT/<配置>/2027"`、`"$RUN_ROOT/<配置>/2028"`，
`--seed` 改为 2027、2028 即可。15 个运行（5 配置 × 3 seed）的原始参数见
`configs/<配置>/<seed>/config.json`。

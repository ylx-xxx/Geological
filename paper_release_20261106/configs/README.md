# 运行配置（15 个运行，脱敏副本）

本目录收录五个正式配置 × 三个 seed 共 15 份 `config.json`。
每份都是对应运行目录中原件的**脱敏副本**：只改路径类字段，不改任何数值超参、seed 与 epoch。

## 1. 目录结构

```
configs/
├── RGB-S1-N0/{2026,2027,2028}/config.json
├── full14-S1-N0/{2026,2027,2028}/config.json
├── RGB-S2-N0/{2026,2027,2028}/config.json
├── full14-S2-N0/{2026,2027,2028}/config.json
├── full14-S2-N2/{2026,2027,2028}/config.json
└── README.md
```

每个 `config.json` 旁都有 `ORIGINAL.json.sha256`，内容为**归档原件**的 SHA256（一行十六进制）。
发布副本自身的 SHA256 见本文件第 4 节，二者不同是正常的：副本已脱敏。

## 2. 被脱敏的字段

| 字段 | 原件 | 副本 |
|---|---|---|
| `data_root` | 服务器数据根目录的绝对路径 | `data/landslide4Sense` |
| `out` | 服务器运行目录的绝对路径 | `runs/<配置>/<seed>` |
| `loveda_checkpoint` | 服务器权重文件绝对路径 | `weights/loveda_best.pth` |
| `initialization_source` | 同上（同一文件的另一字段） | `weights/loveda_best.pth` |
| `terrain_stats` | N2 地形统计文件绝对路径 | `artifacts/terrain_stats/terrain_n2_stats.npz` |
| `terrain_stats_file` | 同上（N2 的另一字段） | `artifacts/terrain_stats/terrain_n2_stats.npz` |
| `argv` | 启动命令中的 3–4 个绝对路径 | 与上表一一对应的相对占位路径 |

补充说明：

- `imagenet_checkpoint` 原件即为空字符串，未改；
- `argv[0]`（训练入口脚本路径）改为 `code_versions/<版本>/reproducibility/train.py`；
- `code_sha256`、`created_utc`、`initial_state_sha256`、`source_checkpoint_sha256`、
  `class_weights`、`train_samples`、`val_samples`、`terrain_stats_sha256` 等字段**原样保留**；
- 历史字段（例如 RGB 配置中的 `"patch_extra": "mean"`）保持原值，没有改写，也不表示本轮启用了均值初始化；
  full14 系列的 `patch_extra` 为 `"zero"`。

## 3. 原件位置与哈希

以下路径相对于冻结归档 `项目离线保全_20261105/`。12 个 N0 运行的 `config_sha256`
与 `最终零训练配对重抽样分析_20261105/inputs/_manifests/freeze_manifest.json` 一致；
3 个 N2 运行的哈希与 `q1_norm_stage1_v1/sha256sums.txt` 一致。

| 配置 | seed | 归档原件 | 原件 SHA256 | 发布副本 SHA256 |
|---|---|---|---|---|
| RGB-S1-N0 | 2026 | `code_snapshot/grsl_work_20261103/lr_screening_20261103/runs/loveda_encoder_lr1e-4_s2026/config.json` | `ca17056bc3cb6cf17e208fe0047086a7ab61c6bd0b52dca7f3ee32a7c86d9023` | `d07ceaae9ffb60b975f8d4115da1013aa7e64488a4cb478b4b8bc509cd8779b6` |
| RGB-S1-N0 | 2027 | `code_snapshot/grsl_work_20261103/lr_screening_20261103/runs/loveda_encoder_lr1e-4_s2027/config.json` | `ff2666ddb4647ec91781aac08e66c61ab07829b9f9b89446814ccb66afc6cfa3` | `d73539dfc4da97ddd4768c7091d439f6c62f44cee8d19feac94cf8d8dc68a578` |
| RGB-S1-N0 | 2028 | `code_snapshot/grsl_work_20261103/lr_screening_20261103/runs/loveda_encoder_lr1e-4_s2028/config.json` | `712adc942ea4d926b396784116a3913a82da7bf797bf676c270c8552775023f8` | `c1cd2055b4ddd4b5f7574f20ae14eb99b4466be201d56637102360b7cb711262` |
| RGB-S2-N0 | 2026 | `code_snapshot/grsl_work_20261103/channel_scale_2x2_20261104/runs/RGB-L-S2-2026/config.json` | `2e6005f64e4d9edfb586365c78216cfbf2b139174ffbcceec19299f0f8c30185` | `f100abc14fdd0fecbf3ae47b9158eaf6cc4350e89525b64ae8b6bb5e2d137c87` |
| RGB-S2-N0 | 2027 | `code_snapshot/grsl_work_20261103/channel_scale_2x2_20261104/runs/RGB-L-S2-2027/config.json` | `d36420a7368800078ba96ee1b8437a2a61e567ae331a69afa3aa030e3b11903f` | `18b40bed61bca9885c3e9fab29b477fd06c2cd03342000c4e625c689eaaa683b` |
| RGB-S2-N0 | 2028 | `code_snapshot/grsl_work_20261103/channel_scale_2x2_20261104/runs/RGB-L-S2-2028/config.json` | `526b28de8bffb514a8cbfb274c04d53fb86163169d6e534a56cc8fd5e9a291af` | `6483136c44000c9c5ebada5c62c709928dacafc350044dcf1f74c81ec6540b84` |
| full14-S1-N0 | 2026 | `code_snapshot/grsl_work_20261103/mc_v1/runs/F14-L-2026/config.json` | `912f7898f037efd7e5b56a38fbbf557864a4696e793df1b155f84aa99a27f945` | `7077a4d05baecc819df12cff7ac6ce5f006d9d53a3a5d34e9b48b2f54b05ba04` |
| full14-S1-N0 | 2027 | `code_snapshot/grsl_work_20261103/mc_v1/runs/F14-L-2027/config.json` | `3834865aa7ac81d75346a1e35b7c9d760a09fb04d793479be91cd826847dd559` | `e9ed926e65f956ec51937741a32fe6d5aabfbb664f003bb553b9c7206c3f75a4` |
| full14-S1-N0 | 2028 | `code_snapshot/grsl_work_20261103/mc_v1/runs/F14-L-2028/config.json` | `9735cbc618f14f92a8c6bd66047a110b29601c52ec05c37a5844e8d78bcabdc8` | `34d0d9aeff35670994de6253f8e2accd0825449fd214b9394db0d0d0dc69931d` |
| full14-S2-N0 | 2026 | `code_snapshot/grsl_work_20261103/scale2_v2/runs/F14-L-S2-2026/config.json` | `11ba76675385de7c12689e4db011e9bd278f03357f94a61aad21df2e5203e5ab` | `fa2d3c6847627dcded3ce5387e82872db2af809ff1d2f27658eeb72cbdf015fe` |
| full14-S2-N0 | 2027 | `code_snapshot/grsl_work_20261103/scale2_v2/runs/F14-L-S2-2027/config.json` | `4d1f7893cb948c8a8e3f8868281795d05037cad858f09f0a8d95a44f8a6bedf3` | `7893f595ab2c8e6e86d8821becb1d4a433ff40b19f6c3d0ec0160724a81a4d53` |
| full14-S2-N0 | 2028 | `code_snapshot/grsl_work_20261103/scale2_v2/runs/F14-L-S2-2028/config.json` | `52df06e6f65e0b29cd25f2af68d147245177f8cccad54d3533076e8b4e92f6e8` | `925f839fa31daf9be97a520fe92272c9b19890d461d71275c81b6de65b188422` |
| full14-S2-N2 | 2026 | `code_snapshot/grsl_work_20261103/q1_norm_stage1_v1/runs/F14-L-S2-N2-2026/config.json` | `876701272740ed2469985b9cf756ba16dda356b2fc6c72b68aa25086b269d5d5` | `55db07373e4c70cac3d61d78fb5db7eb17188781a1ee1f38a49aafee01f75cd9` |
| full14-S2-N2 | 2027 | `code_snapshot/grsl_work_20261103/q1_norm_stage1_v1/runs/F14-L-S2-N2-2027/config.json` | `44fa8f408717611cc07a7c542f14a31ed7e7b5179002bd3ba960d573b76b5c6a` | `d4d0f2e3e34518bfbc98dbee7f159f695c72a5a826b4ced3bd921ac2a5eba757` |
| full14-S2-N2 | 2028 | `code_snapshot/grsl_work_20261103/q1_norm_stage1_v1/runs/F14-L-S2-N2-2028/config.json` | `cb128c780b8d401994c3f80a3bb62b069346467ed4ec8a03198e4658c7c53913` | `7351ba1b091be8450efb3807c67f2e6f1c2414e598c755551b812cdbc5840909` |

## 4. 字段差异自查

15 份副本与原件逐字段比较后，只有第 2 节列出的路径字段不同；其余字段（含全部数值超参）
完全相同。副本中的 `out` 与 `argv` 同时改变，二者保持一致；`argv` 中的数值参数顺序与原件相同。

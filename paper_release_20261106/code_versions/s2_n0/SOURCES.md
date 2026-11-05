# 代码来源（s2_n0）

对应正式配置：RGB-S2-N0、full14-S2-N0。

来源归档：`项目离线保全_20261105/code_snapshot/grsl_work_20261103/code_scale2_v2/`。

服务器上的原始绝对路径刻意不写入本文件，以免发布副本含服务器路径；上述归档相对路径是本次核对的来源依据。

版本说明：S2 是模型内部 128→256→128 的尺度包装器，不是新的观测分辨率。包含 `scaled_model.py` 与评价入口 `eval_scale2.py`；`stca.py` 仅为训练入口的静态导入所需，STCA 在正式配置中未启用。

## 1. 文件清单

下表路径均相对本版本目录 `code_versions/s2_n0/`。`原件 SHA256` 为归档中原件的哈希；
`副本 SHA256` 为写入本仓库后的哈希。

| 文件 | 归档原件相对路径 | 原件 SHA256 | 副本 SHA256 | 说明 |
|---|---|---|---|---|
| `reproducibility/__init__.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/reproducibility/__init__.py` | `28fa8ee60022de8cd29345aeb80e4e1fe5a8a0c76c324af760525fae7e567f05` | `28fa8ee60022de8cd29345aeb80e4e1fe5a8a0c76c324af760525fae7e567f05` | 逐字节副本 |
| `reproducibility/audit_dataset.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/reproducibility/audit_dataset.py` | `362ce197dd47660711a761e8a5ca89ac6159edeefb4961124381222acd096eee` | `362ce197dd47660711a761e8a5ca89ac6159edeefb4961124381222acd096eee` | 逐字节副本 |
| `reproducibility/core.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/reproducibility/core.py` | `94de11e0e68333870f2c41eea662ccf0d7d8e7a635584a0a2c4f7ff4cdd255d4` | `94de11e0e68333870f2c41eea662ccf0d7d8e7a635584a0a2c4f7ff4cdd255d4` | 逐字节副本 |
| `reproducibility/eval_scale2.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/reproducibility/eval_scale2.py` | `ae03f192b97115e7d01e2f95951090bd9f2958730c529e5ea4b633c4ccf07dcc` | `ae03f192b97115e7d01e2f95951090bd9f2958730c529e5ea4b633c4ccf07dcc` | 逐字节副本 |
| `reproducibility/evaluate.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/reproducibility/evaluate.py` | `83e54d36417b02eb5e531b2a6738e037d130f0a1d50157b5088405287f0623fc` | `83e54d36417b02eb5e531b2a6738e037d130f0a1d50157b5088405287f0623fc` | 逐字节副本 |
| `reproducibility/train.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/reproducibility/train.py` | `c41b4acfd4be8581979e60604106a65641bdd1637deb90f4600d9c8472197dda` | `c41b4acfd4be8581979e60604106a65641bdd1637deb90f4600d9c8472197dda` | 逐字节副本 |
| `tests/test_build_parity.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/tests/test_build_parity.py` | `2535271c1c282cf54ab35bed9f2e7610e237a253216631933f1c1e6f573cac1b` | `2535271c1c282cf54ab35bed9f2e7610e237a253216631933f1c1e6f573cac1b` | 逐字节副本 |
| `tests/test_patch_mapping.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/tests/test_patch_mapping.py` | `4f7b7efb6422814540e1cbb364dfebf29e144008153349c97d3e89c138f80a4f` | `4f7b7efb6422814540e1cbb364dfebf29e144008153349c97d3e89c138f80a4f` | 逐字节副本 |
| `tests/test_reproducibility.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/tests/test_reproducibility.py` | `20b667ee62246d03f9aaccc231e0206d6ba0b89f86f502f3144f79504c489f09` | `20b667ee62246d03f9aaccc231e0206d6ba0b89f86f502f3144f79504c489f09` | 逐字节副本 |
| `train_l4s_qz/check_l4s.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/check_l4s.py` | `5d89224807ca84080327e6c430a46f3815705df222cf3b34ba1967d26089c743` | `24a168aa81566ba0d6146af5856d4ae0c97c12e3c33d0b5f037d17804dae1be9` | 脱敏副本（见第 2 节） |
| `train_l4s_qz/dataset.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/dataset.py` | `3bd5f65bf99d99c2da81060a42216404c21f2fccd185c9dd6ee27d3aad3d7735` | `3bd5f65bf99d99c2da81060a42216404c21f2fccd185c9dd6ee27d3aad3d7735` | 逐字节副本 |
| `train_l4s_qz/losses.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/losses.py` | `b5795570dde134cfcedd520db71024ad7dd9a381717a257d501eec81c188a11d` | `b5795570dde134cfcedd520db71024ad7dd9a381717a257d501eec81c188a11d` | 逐字节副本 |
| `train_l4s_qz/metrics.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/metrics.py` | `ef1057dcfcc1c125a26bb84485df2722fcbeba805353c6c9d8b2df80a9f5dddf` | `ef1057dcfcc1c125a26bb84485df2722fcbeba805353c6c9d8b2df80a9f5dddf` | 逐字节副本 |
| `train_l4s_qz/model.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/model.py` | `e700d3f616234b1a118aad7ae528ad322b2a431fa600652ea8dc31a085437f93` | `e700d3f616234b1a118aad7ae528ad322b2a431fa600652ea8dc31a085437f93` | 逐字节副本 |
| `train_l4s_qz/scaled_model.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/scaled_model.py` | `9844af585a2e4f1f588f815a03aed6e54c1237eff6478a747e62cba2adcc5855` | `9844af585a2e4f1f588f815a03aed6e54c1237eff6478a747e62cba2adcc5855` | 逐字节副本 |
| `train_l4s_qz/stca.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/stca.py` | `7ffe51e1ef61a2a7e43d16dbd0d960a47ffb25e3e6b9269990434db7c3a4a30c` | `7ffe51e1ef61a2a7e43d16dbd0d960a47ffb25e3e6b9269990434db7c3a4a30c` | 逐字节副本 |
| `train_l4s_qz/test.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/test.py` | `edee37b81e10d32e9f343e282171dcc6ff2940b68391228f58dfa6e4f372edae` | `6f24cf5f670d534d336b5fe50e2130ee2434a9d0db30c6f76ec1d6d7d0ac6bd9` | 脱敏副本（见第 2 节） |
| `train_l4s_qz/time_budget.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/time_budget.py` | `f37613d27504efc67556e54c3b7c977085c88e046a3b8dc0ec335afc4fa3c22e` | `f37613d27504efc67556e54c3b7c977085c88e046a3b8dc0ec335afc4fa3c22e` | 逐字节副本 |
| `train_l4s_qz/train.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/train.py` | `dfb8f869a690bf5b8ae7c5c352a5d616c3a2ba613b52cd42435dce6d8cd4e163` | `aab382877c9efaf00b97a2170a0f189c78c8d7a14774b34bfb6c7b86ab1c1a28` | 脱敏副本（见第 2 节） |
| `train_l4s_qz/utils.py` | `code_snapshot/grsl_work_20261103/code_scale2_v2/train_l4s_qz/utils.py` | `37231aa43f69e109bb2d9d4abfe6e7ed25f1b662406c82f0eb7c1da6592e7f48` | `37231aa43f69e109bb2d9d4abfe6e7ed25f1b662406c82f0eb7c1da6592e7f48` | 逐字节副本 |

## 2. 逐字节声明与脱敏例外

除上表标注为“脱敏副本”的文件外，其余文件均为**逐字节副本**：内容与归档原件完全相同，
副本 SHA256 与原件 SHA256 一致，可直接用哈希核对。

脱敏副本的处理：这些文件在 `__main__` 的 argparse 默认值或模块级常量中硬编码了
服务器绝对路径。为保证发布副本不含服务器路径，按下列对应关系做了最小替换：

- `train_l4s_qz/check_l4s.py`：数据根目录（绝对路径）→ `data/landslide4Sense`
- `train_l4s_qz/test.py`：数据根目录（绝对路径）→ `data/landslide4Sense`；日志目录（绝对路径）→ `logs/train_l4s_qz`
- `train_l4s_qz/train.py`：数据根目录（绝对路径）→ `data/landslide4Sense`；LoveDA 源权重（绝对路径）→ `weights/loveda_best.pth`；checkpoint 输出目录（绝对路径）→ `runs/train_l4s_qz`；日志目录（绝对路径）→ `logs/train_l4s_qz`

替换只影响缺省值；训练与评价实际运行时均由显式参数传入路径。类名、函数名、
算法与数值逻辑均未改动，也未新增或删除任何代码行。逐文件的替换清单见
`provenance/release_source_map.csv`。

## 3. 未收录的模块

`bench_penalty.py`、`calibrate_lambda.py`、`postprocess_search.py`、`threshold_search.py`、`plot_diagnostics.py`、`audit_sen12_subset.py`、`prepare_sen12_subset.py`（区域 Sen12 子集准备，非本任务数据），以及 `test_empty_penalty.py`、`test_time_budget.py`、`test_stca.py`、`test_sen12_preparation.py`（对应未启用分支或无关数据）。其中 `postprocess_search.py`、`threshold_search.py` 未被训练/评价入口 import；`time_budget.py` 被训练入口 import，因此模块保留。

## 4. 复现入口

```bash
cd code_versions/s2_n0
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -m reproducibility.train --help
```

训练与评价命令见 `commands/train.md`、`commands/evaluate.md`。

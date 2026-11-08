# 代码来源（rgb_s1）

对应正式配置：RGB-S1-N0。

来源归档：`项目离线保全_20261105/code_snapshot/grsl_work_20261103/code/`。

服务器上的原始绝对路径刻意不写入本文件，以免发布副本含服务器路径；上述归档相对路径是本次核对的来源依据。

版本说明：物理 RGB、原生 128 输入、N0 归一化。不含 S2 尺度包装器、STCA、时间上限等模块（该版本训练入口不引用它们）。

## 1. 文件清单

下表路径均相对本版本目录 `code_versions/rgb_s1/`。`原件 SHA256` 为归档中原件的哈希；
`副本 SHA256` 为写入本仓库后的哈希。

| 文件 | 归档原件相对路径 | 原件 SHA256 | 副本 SHA256 | 说明 |
|---|---|---|---|---|
| `reproducibility/__init__.py` | `code_snapshot/grsl_work_20261103/code/reproducibility/__init__.py` | `28fa8ee60022de8cd29345aeb80e4e1fe5a8a0c76c324af760525fae7e567f05` | `28fa8ee60022de8cd29345aeb80e4e1fe5a8a0c76c324af760525fae7e567f05` | 逐字节副本 |
| `reproducibility/audit_dataset.py` | `code_snapshot/grsl_work_20261103/code/reproducibility/audit_dataset.py` | `362ce197dd47660711a761e8a5ca89ac6159edeefb4961124381222acd096eee` | `362ce197dd47660711a761e8a5ca89ac6159edeefb4961124381222acd096eee` | 逐字节副本 |
| `reproducibility/core.py` | `code_snapshot/grsl_work_20261103/code/reproducibility/core.py` | `94de11e0e68333870f2c41eea662ccf0d7d8e7a635584a0a2c4f7ff4cdd255d4` | `94de11e0e68333870f2c41eea662ccf0d7d8e7a635584a0a2c4f7ff4cdd255d4` | 逐字节副本 |
| `reproducibility/evaluate.py` | `code_snapshot/grsl_work_20261103/code/reproducibility/evaluate.py` | `83e54d36417b02eb5e531b2a6738e037d130f0a1d50157b5088405287f0623fc` | `83e54d36417b02eb5e531b2a6738e037d130f0a1d50157b5088405287f0623fc` | 逐字节副本 |
| `reproducibility/train.py` | `code_snapshot/grsl_work_20261103/code/reproducibility/train.py` | `64f8c4bccca034653f3b04bf997c25115172a98cdc4a87d2b23c9a585fc3c3f4` | `64f8c4bccca034653f3b04bf997c25115172a98cdc4a87d2b23c9a585fc3c3f4` | 逐字节副本 |
| `tests/test_reproducibility.py` | `code_snapshot/grsl_work_20261103/code/tests/test_reproducibility.py` | `20b667ee62246d03f9aaccc231e0206d6ba0b89f86f502f3144f79504c489f09` | `20b667ee62246d03f9aaccc231e0206d6ba0b89f86f502f3144f79504c489f09` | 逐字节副本 |
| `train_l4s_qz/check_l4s.py` | `code_snapshot/grsl_work_20261103/code/train_l4s_qz/check_l4s.py` | `5d89224807ca84080327e6c430a46f3815705df222cf3b34ba1967d26089c743` | `24a168aa81566ba0d6146af5856d4ae0c97c12e3c33d0b5f037d17804dae1be9` | 脱敏副本（见第 2 节） |
| `train_l4s_qz/dataset.py` | `code_snapshot/grsl_work_20261103/code/train_l4s_qz/dataset.py` | `3bd5f65bf99d99c2da81060a42216404c21f2fccd185c9dd6ee27d3aad3d7735` | `3bd5f65bf99d99c2da81060a42216404c21f2fccd185c9dd6ee27d3aad3d7735` | 逐字节副本 |
| `train_l4s_qz/losses.py` | `code_snapshot/grsl_work_20261103/code/train_l4s_qz/losses.py` | `fd34d4674d799c19b9a9c6a3c1bf61779edb62421d0adf903cc04cb4b4e2b1ba` | `fd34d4674d799c19b9a9c6a3c1bf61779edb62421d0adf903cc04cb4b4e2b1ba` | 逐字节副本 |
| `train_l4s_qz/metrics.py` | `code_snapshot/grsl_work_20261103/code/train_l4s_qz/metrics.py` | `ef1057dcfcc1c125a26bb84485df2722fcbeba805353c6c9d8b2df80a9f5dddf` | `ef1057dcfcc1c125a26bb84485df2722fcbeba805353c6c9d8b2df80a9f5dddf` | 逐字节副本 |
| `train_l4s_qz/model.py` | `code_snapshot/grsl_work_20261103/code/train_l4s_qz/model.py` | `8d91a7b20f08a2d2e9370a9ed21c9ef78d46555a6fb5ebc319acd0fd9b667955` | `8d91a7b20f08a2d2e9370a9ed21c9ef78d46555a6fb5ebc319acd0fd9b667955` | 逐字节副本 |
| `train_l4s_qz/test.py` | `code_snapshot/grsl_work_20261103/code/train_l4s_qz/test.py` | `edee37b81e10d32e9f343e282171dcc6ff2940b68391228f58dfa6e4f372edae` | `6f24cf5f670d534d336b5fe50e2130ee2434a9d0db30c6f76ec1d6d7d0ac6bd9` | 脱敏副本（见第 2 节） |
| `train_l4s_qz/train.py` | `code_snapshot/grsl_work_20261103/code/train_l4s_qz/train.py` | `dfb8f869a690bf5b8ae7c5c352a5d616c3a2ba613b52cd42435dce6d8cd4e163` | `aab382877c9efaf00b97a2170a0f189c78c8d7a14774b34bfb6c7b86ab1c1a28` | 脱敏副本（见第 2 节） |
| `train_l4s_qz/utils.py` | `code_snapshot/grsl_work_20261103/code/train_l4s_qz/utils.py` | `37231aa43f69e109bb2d9d4abfe6e7ed25f1b662406c82f0eb7c1da6592e7f48` | `37231aa43f69e109bb2d9d4abfe6e7ed25f1b662406c82f0eb7c1da6592e7f48` | 逐字节副本 |

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

本版本源码目录中与本次五配置结论无关、未复制的文件：`postprocess_search.py`、`threshold_search.py`
（均未被训练/评价入口 import）、`plot_diagnostics.py`（绘图辅助）、`audit_sen12_subset.py` 与
`prepare_sen12_subset.py`（区域 Sen12 子集准备，非本任务数据），以及 `test_sen12_preparation.py`
（对应上述无关数据）。

## 4. 复现入口

```bash
cd code_versions/rgb_s1
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -m reproducibility.train --help
```

训练与评价命令见 `commands/train.md`、`commands/evaluate.md`。

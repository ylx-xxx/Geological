# 溯源与冻结清单

本目录保存原分析的输入清单、检查点/测试冻结记录，以及发布文件的逐文件来源映射。

## 文件

| 文件 | 内容 |
|---|---|
| `input_manifest.json` | 30 组 `per_sample.csv` / `protocol.json` / `metrics.json` 的原始 SHA256、样本数、patch 平均 IoU、pooled IoU；bootstrap 定义；配置与种子；split 规模；Test 复用限制说明。 |
| `freeze_manifest.json` | 各配置/种子的检查点冻结记录：权重路径、SHA256、训练/评价目录、关键指标。 |
| `test_manifest.json` | 历史 Test 评价的执行记录：命令、输入、检查点、输出目录与逐文件哈希。 |
| `q1_frozen_bests.json` | N2 配置三个种子的 best-Valid epoch、检查点 SHA256、选模指标。 |
| `manifest_index.json` | 上述三份冻结清单的原始远端位置与 SHA256。 |
| `release_source_map_results.csv` | 本次发布写入的每个文件一行：发布路径、归档相对源路径、源 SHA256、是否脱敏派生、备注。 |

## 脱敏处理

原清单与 6 个 N2 `protocol.json` 含有服务器绝对路径，本次发布做了**仅路径字符串**的脱敏：

| 原路径类别 | 替换为 | 出现范围 |
|---|---|---|
| 云端数据根目录（原绝对路径前缀） | `<REMOTE_DATA_ROOT>` | 5 份清单 + 6 个 N2 `protocol.json` |
| Python 环境根目录（原绝对路径前缀） | `<REMOTE_ENV_ROOT>` | `test_manifest.json` |

- 未发现个人用户名、邮箱、本机 Windows 路径或作者信息。
- 脱敏只改路径前缀，**清单内的所有 SHA256、epoch、指标、命令参数与数值均未改动**，JSON 结构保持有效。
- 原有哈希未被重算或替换。由于脱敏文件的字节已变化，其**发布副本**的 SHA256 与清单中记录的**原始文件** SHA256 不同；原始 SHA256 仍原样保留，用于核对源归档，而不是核对脱敏副本。
- 受影响的 6 个 N2 `protocol.json` 位于 `../results/inputs/{valid,test}__F14-S2-N2__{2026,2027,2028}/protocol.json`；除 `terrain_stats_file` 外，其余字段与源文件逐字段一致。其余 84 个结果数据文件未脱敏、逐字节一致。
- `release_source_map_results.csv` 中 `derived=true` 精确标记了上述 11 个脱敏文件（5 份清单 + 6 个 N2 协议）；其余文件为 `false`。

## 来源映射表列

- `release_path`：相对仓库根目录的发布路径。
- `source_path`：相对论文材料归档根目录的源路径，前缀为 `最终零训练配对重抽样分析_20261105/` 或 `项目离线保全_20261105/`。
- `source_sha256`：源文件（脱敏前）的 SHA256；本次撰写的说明文件留空。
- `derived`：`true` 表示发布副本经过脱敏等派生处理；`false` 表示与源文件逐字节一致（或为本任务撰写的说明文件）。
- `note`：处理说明。

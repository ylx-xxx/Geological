# 历史材料归档

本目录保存 2026-11-06 受控研究发布之前的实现、实验记录、图件与文档，仅用于追溯历史身份，**不代表当前发布结论**。

当前发布入口见仓库根目录 [`README.md`](../README.md) 与 [`PROTOCOL.md`](../PROTOCOL.md)。

## 目录说明

| 路径 | 内容 |
|---|---|
| `docs/` | 历史实验协议、复现说明与材料状态审计 |
| `q2_experiments/` | 历史论文表格与图件；各行训练协议不同，不能当作同一基线 |
| `q2_extra_ablation/` | 历史消融与 Sen12 探索性结果 |
| `reproducibility/` | 历史审计评估、数据审计与通道修正训练代码 |
| `tests/` | 上述历史代码的单元测试 |
| `train_l4s_qz/`、`train_l4s_qz_v2/` … `train_l4s_qz_v8_ft/` | 历史训练实现版本，保留开发过程 |
| `train_la_qz/` | 更早的历史训练实现 |
| `requirements-audit.txt` | 2026-10-04 观测到的运行环境 |
| `old流程图.png` | 旧版流程图 |

## 使用注意

- 历史代码按原顶层包名组织（`reproducibility/`、`train_l4s_qz/` 等），因此相关命令需在 `archive/` 目录下执行，例如：
  ```bash
  cd archive
  python -m unittest discover -s tests -v
  ```
- 历史结果（0.4793 / 0.4426 / 0.4231 等）与当前发布**互不替代**，各自保留原有协议身份。
- 旧 Sen12 材料保持探索性身份，不作为当前发布的独立外部验证。
- 本目录只做位置归档，历史数值、配置与协议内容未改动。

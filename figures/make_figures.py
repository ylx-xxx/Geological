# -*- coding: utf-8 -*-
"""生成 G1–G3 固定图（只描述已完成实验，不重算指标、不改阈值）。

用法：
  python -I -X utf8 make_figures.py --tables <最终零训练配对重抽样分析目录> \
      --snapshot <项目离线保全/code_snapshot> --out .

数据来源：
  G1/G2：既有逐样本评价表与 bootstrap_summary.csv（本机 CPU 计算，未重新推理）。
  G3：既有 60 轮 history.csv（按配置与种子实际训练目录读取）。
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 五个配置的固定顺序与配色（Okabe–Ito 色盲友好方案）
CONFIGS = ["RGB-S1", "F14-S1", "RGB-S2", "F14-S2", "F14-S2-N2"]
LABELS = {
    "RGB-S1": "RGB-S1-N0",
    "F14-S1": "full14-S1-N0",
    "RGB-S2": "RGB-S2-N0",
    "F14-S2": "full14-S2-N0",
    "F14-S2-N2": "full14-S2-N2",
}
COLORS = {
    "RGB-S1": "#0072B2",
    "F14-S1": "#E69F00",
    "RGB-S2": "#009E73",
    "F14-S2": "#D55E00",
    "F14-S2-N2": "#CC79A7",
}
MARKERS = {"RGB-S1": "o", "F14-S1": "s", "RGB-S2": "^", "F14-S2": "D", "F14-S2-N2": "v"}
SEEDS = ["2026", "2027", "2028"]
CONTRASTS = [("C1", "full14-S2 − RGB-S2", "multi-channel"),
             ("C2", "full14-S2-N2 − full14-S2", "terrain normalization"),
             ("R1", "RGB-S2 − RGB-S1", "RGB scale"),
             ("R2", "full14-S2 − full14-S1", "full14 scale")]


def read_pooled_iou(table_root, split, cfg, seed):
    """由逐样本表重算 pooled IoU（先求和再相除）。"""
    p = os.path.join(table_root, f"{split}__{cfg}__{seed}", "per_sample.csv")
    tp = fp = fn = 0
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        tp += int(r["TP"]); fp += int(r["FP"]); fn += int(r["FN"])
    return tp / (tp + fp + fn)


def load_g1(table_root):
    """返回每配置的 Valid/Test 三种子 IoU。"""
    data = {}
    for cfg in CONFIGS:
        data[cfg] = {}
        for split in ["valid", "test"]:
            data[cfg][split] = [read_pooled_iou(table_root, split, cfg, s) for s in SEEDS]
    return data


def fig_g1(data, out):
    """G1：五配置 Valid/Test 绝对性能，点 = 每 seed，均值 ± 样本 SD。"""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(CONFIGS))
    offsets = {"valid": -0.17, "test": 0.17}
    for split, name in [("valid", "Valid (245)"), ("test", "Test (800, historical)")]:
        means = [np.mean(data[c][split]) for c in CONFIGS]
        sds = [np.std(data[c][split], ddof=1) for c in CONFIGS]
        ax.errorbar(x + offsets[split], means, yerr=sds, fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3, zorder=2)
        for i, c in enumerate(CONFIGS):
            # Valid 空心、Test 实心；两者都用同一配置色，跨图配色一致
            fc = "white" if split == "valid" else COLORS[c]
            ec = COLORS[c] if split == "valid" else "black"
            ax.scatter([x[i] + offsets[split]] * 3, data[c][split],
                       s=24, marker="o", facecolors=fc,
                       edgecolors=ec, linewidths=0.9, zorder=3)
            ax.scatter([x[i] + offsets[split]], [means[i]], s=70, marker="_",
                       color="black", linewidths=1.6, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONFIGS], rotation=15, ha="right")
    ax.set_ylabel("Landslide IoU")
    ax.set_ylim(0.0, 0.65)  # 从 0 起，不截断坐标夸大差异
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    # 图例：空心 = Valid，实心 = Test
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markeredgecolor="black",
                      markerfacecolor="white", markersize=6, label="Valid mean ± SD (3 seeds)"),
               Line2D([0], [0], marker="o", color="none", markeredgecolor="black",
                      markerfacecolor="black", markersize=6, label="Test mean ± SD (3 seeds)")]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8)
    ax.set_title("G1  Absolute performance at frozen checkpoints", fontsize=10)
    fig.tight_layout()
    save(fig, os.path.join(out, "G1_absolute_iou"))


def fig_g2(summary_path, out):
    """G2：四对比的 ΔValid、ΔTest、D 点估计与 95% 百分位区间。"""
    rows = {r["contrast_id"]: r for r in csv.DictReader(open(summary_path, encoding="utf-8"))}
    panels = [("delta_valid", "ΔValid  (IoU percentage points)"),
              ("delta_test", "ΔTest  (IoU percentage points)"),
              ("D", "D = ΔTest − ΔValid  (IoU percentage points)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), sharey=True)
    y = np.arange(len(CONTRASTS))[::-1]
    for ax, (key, title) in zip(axes, panels):
        for yi, (cid, label, role) in zip(y, CONTRASTS):
            r = rows[cid]
            p = float(r[f"{key}_point"]) * 100
            lo = float(r[f"{key}_lo"]) * 100
            hi = float(r[f"{key}_hi"]) * 100
            ax.errorbar(p, yi, xerr=[[p - lo], [hi - p]], fmt="o", color="#1F3A5F",
                        markersize=5, elinewidth=1.3, capsize=3, zorder=3)
        ax.axvline(0, color="black", linewidth=0.9, linestyle="-", zorder=1)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{cid}: {label}" for cid, label, _ in CONTRASTS], fontsize=8)
    fig.suptitle("G2  Paired patch-bootstrap effects (5000 replicates; 95% percentile intervals)",
                 fontsize=10)
    fig.text(0.5, 0.01,
             "Frozen checkpoints; patches treated as exchangeable; no spatial correction; "
             "Test is a historical benchmark already used during research feedback.\n"
             "Intervals crossing zero are not distinguishable from zero; point estimates are shown regardless.",
             ha="center", fontsize=7, color="0.25")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    save(fig, os.path.join(out, "G2_paired_effects"))


def history_path(snapshot, cfg, seed):
    """按冻结/实际运行记录定位 history.csv。"""
    rel = {
        "RGB-S1": f"grsl_work_20261103/lr_screening_20261103/runs/loveda_encoder_lr1e-4_s{seed}",
        "F14-S1": f"grsl_work_20261103/mc_v1/runs/F14-L-{seed}",
        "RGB-S2": f"grsl_work_20261103/channel_scale_2x2_20261104/runs/RGB-L-S2-{seed}",
        "F14-S2": f"grsl_work_20261103/scale2_v2/runs/F14-L-S2-{seed}",
        "F14-S2-N2": f"grsl_work_20261103/q1_norm_stage1_v1/runs/F14-L-S2-N2-{seed}",
    }[cfg]
    return os.path.join(snapshot, rel, "history.csv")


def fig_g3(snapshot, out):
    """G3：三个种子分面的 60 轮 Valid IoU 曲线与选模 epoch。"""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for cfg in CONFIGS:
            p = history_path(snapshot, cfg, seed)
            if not os.path.exists(p):
                ax.text(0.5, 0.5, f"missing: {cfg}", transform=ax.transAxes,
                        ha="center", fontsize=7, color="red")
                continue
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            ep = np.array([int(r["epoch"]) for r in rows])
            iou = np.array([float(r["val_landslide_iou"]) for r in rows])
            ax.plot(ep, iou, color=COLORS[cfg], linewidth=1.1, label=LABELS[cfg])
            bi = int(np.argmax(iou))
            ax.scatter(ep[bi], iou[bi], color=COLORS[cfg], marker="*", s=70,
                       edgecolors="black", linewidths=0.4, zorder=4)
        ax.set_title(f"seed {seed}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(color="0.92", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Valid Landslide IoU")
    axes[0].set_ylim(0.0, 0.65)
    axes[2].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("G3  Training curves at 60 epochs; stars mark the best-Valid epoch used for selection",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, os.path.join(out, "G3_training_curves"))


def save(fig, stem):
    """统一导出矢量 + 300dpi PNG。"""
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print("wrote", stem + ".{pdf,svg,png}")


def write_tables(data, summary_path, out):
    """导出论文用主表与效应表（只引用既有真实指标）。"""
    with open(os.path.join(out, "paper_main_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "valid_iou_seed2026", "valid_iou_seed2027", "valid_iou_seed2028",
                    "valid_mean", "valid_sd", "test_iou_seed2026", "test_iou_seed2027",
                    "test_iou_seed2028", "test_mean", "test_sd"])
        for c in CONFIGS:
            v = data[c]["valid"]; t = data[c]["test"]
            w.writerow([LABELS[c]] + [f"{x:.6f}" for x in v] +
                       [f"{np.mean(v):.6f}", f"{np.std(v, ddof=1):.6f}"] +
                       [f"{x:.6f}" for x in t] +
                       [f"{np.mean(t):.6f}", f"{np.std(t, ddof=1):.6f}"])
    rows = list(csv.DictReader(open(summary_path, encoding="utf-8")))
    fields = ["contrast_id", "contrast", "delta_valid_point", "delta_valid_lo", "delta_valid_hi",
              "delta_test_point", "delta_test_lo", "delta_test_hi",
              "D_point", "D_lo", "D_hi", "n_reps", "invalid_reps",
              "seeds_delta_valid_positive", "seeds_delta_test_positive", "seeds_D_positive"]
    with open(os.path.join(out, "paper_effects_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote paper_main_table.csv, paper_effects_table.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True,
                    help="含 inputs/{split}__{cfg}__{seed}/per_sample.csv 与 bootstrap_summary.csv 的目录")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load_g1(os.path.join(a.tables, "inputs"))
    fig_g1(data, a.out)
    fig_g2(os.path.join(a.tables, "bootstrap_summary.csv"), a.out)
    fig_g3(a.snapshot, a.out)
    write_tables(data, os.path.join(a.tables, "bootstrap_summary.csv"), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

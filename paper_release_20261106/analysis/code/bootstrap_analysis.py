# -*- coding: utf-8 -*-
"""最终零训练配对 patch 重抽样分析（CPU / NumPy）。

严格按任务书第 4 节：
- B=5000，随机种子 20261105；
- 同一 split、同一 replicate 内所有配置与种子共用同一组 patch 索引；
- Valid 与 Test 分别生成抽样索引；
- 每个 seed/config 先对混淆计数求和再算 pooled IoU，然后同 seed 做差，最后对三个 seed 取均值；
- 比较量 delta_valid / delta_test / D 的原始点估计与百分位 2.5%/97.5% 区间。

本脚本只读取已导出的逐样本评价表，不做任何模型推理。
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# 五个唯一配置的展示顺序（分析只涉及这五个）
CONFIGS = ["RGB-S1", "F14-S1", "RGB-S2", "F14-S2", "F14-S2-N2"]
SEEDS = ["2026", "2027", "2028"]
SPLITS = ["valid", "test"]

# 四个固定对比，不事后增删
CONTRASTS = [
    # (ID, 左配置, 右配置, 角色)
    ("C1", "F14-S2", "RGB-S2", "main: multi-channel effect"),
    ("C2", "F14-S2-N2", "F14-S2", "secondary: terrain normalization effect"),
    ("R1", "RGB-S2", "RGB-S1", "reference: RGB scale effect"),
    ("R2", "F14-S2", "F14-S1", "reference: full14 scale effect"),
]

B_REPS = 5000
BASE_SEED = 20261105


def sha256_file(path):
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_dir(input_root, split, config, seed):
    """按导出命名规则定位单个逻辑评价表的目录。"""
    return os.path.join(input_root, f"{split}__{config}__{seed}")


def load_table(path):
    """读取逐样本表，返回 sample 顺序、混淆计数矩阵与 patch 级指标。"""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    samples = [r["sample"] for r in rows]
    counts = {k: np.array([int(r[k]) for r in rows], dtype=np.int64)
              for k in ["positive_pixels", "valid_pixels", "TN", "FP", "FN", "TP"]}
    patch_iou = np.array([float(r["Landslide_IoU"]) for r in rows], dtype=np.float64)
    return {"samples": samples, "counts": counts, "patch_iou": patch_iou}


def load_split(input_root, split):
    """载入一个 split 的 15 份表（5 配置 × 3 种子），显式按 sample 对齐。

    返回：
      sample_order: 基准样本顺序
      TP/FP/FN: 形状 (3 seed, 5 config, n_sample) 的数组
      meta: 每份表的来源与哈希信息
    """
    raw = {}
    for cfg in CONFIGS:
        for seed in SEEDS:
            d = table_dir(input_root, split, cfg, seed)
            p = os.path.join(d, "per_sample.csv")
            raw[(cfg, seed)] = load_table(p)

    # 样本身份：以第一个配置/bootstrap 种子表为基准，其余显式按 sample 重排
    ref_cfg, ref_seed = CONFIGS[0], SEEDS[0]
    sample_order = raw[(ref_cfg, ref_seed)]["samples"]
    if len(set(sample_order)) != len(sample_order):
        raise ValueError(f"{split}: 基准表 sample 存在重复")

    n = len(sample_order)
    for key, t in raw.items():
        if len(t["samples"]) != n or set(t["samples"]) != set(sample_order):
            raise ValueError(f"{split}/{key}: 样本集合与基准不一致，拒绝静默丢弃")
        pos = {s: i for i, s in enumerate(t["samples"])}
        order = [pos[s] for s in sample_order]  # 把该表重排到基准顺序
        t["aligned"] = {k: v[order] for k, v in t["counts"].items()}
        # 同一 split 内标签应完全一致
    for key, t in raw.items():
        a = t["aligned"]
        ref = raw[(ref_cfg, ref_seed)]["aligned"]
        if not np.array_equal(a["positive_pixels"], ref["positive_pixels"]):
            raise ValueError(f"{split}/{key}: positive_pixels 与其他表不一致")
        if not np.array_equal(a["valid_pixels"], ref["valid_pixels"]):
            raise ValueError(f"{split}/{key}: valid_pixels 与其他表不一致")

    TP = np.stack([np.stack([raw[(cfg, seed)]["aligned"]["TP"] for cfg in CONFIGS])
                   for seed in SEEDS])  # (3, 5, n)
    FP = np.stack([np.stack([raw[(cfg, seed)]["aligned"]["FP"] for cfg in CONFIGS])
                   for seed in SEEDS])
    FN = np.stack([np.stack([raw[(cfg, seed)]["aligned"]["FN"] for cfg in CONFIGS])
                   for seed in SEEDS])

    meta = {}
    for (cfg, seed), t in raw.items():
        d = table_dir(input_root, split, cfg, seed)
        meta[f"{split}|{cfg}|{seed}"] = {
            "local_dir": d,
            "per_sample_csv_sha256": sha256_file(os.path.join(d, "per_sample.csv")),
            "protocol_sha256": sha256_file(os.path.join(d, "protocol.json")),
            "metrics_sha256": sha256_file(os.path.join(d, "metrics.json")),
            "n_samples": n,
            "patch_mean_iou": float(t["patch_iou"].mean()),
            "pooled_iou_full": None,  # 稍后填
        }
    return sample_order, TP, FP, FN, meta


def pooled_iou(tp, fp, fn):
    """由混淆计数求和计算 pooled IoU；分母为零返回 NaN，不补零。"""
    denom = tp + fp + fn
    out = np.full(np.shape(tp), np.nan, dtype=np.float64)
    np.divide(tp, denom, out=out, where=denom > 0)
    return out


def bootstrap_weights(rng, n, reps):
    """生成 (reps, n) 的 patch 重抽样多重性矩阵。

    每行由 n 次有放回抽取的频数构成，可直接与逐 patch 混淆计数做矩阵乘法。
    """
    w = np.zeros((reps, n), dtype=np.int32)
    for b in range(reps):
        idx = rng.integers(0, n, size=n)
        w[b] = np.bincount(idx, minlength=n)
    return w


def summarize(arr, name):
    """百分位区间与基本统计。"""
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"point": None, "lo": float(lo), "hi": float(hi),
            "mean_reps": float(np.mean(arr)), "sd_reps": float(np.std(arr, ddof=1)),
            "n_reps": int(arr.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reps", type=int, default=B_REPS)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = {}
    meta = {}
    for split in SPLITS:
        sample_order, TP, FP, FN, m = load_split(args.input_root, split)
        data[split] = {"n": len(sample_order), "TP": TP, "FP": FP, "FN": FN}
        meta.update(m)

    # 全量（未重抽样）点估计，可直接与既有汇总对照
    full_iou = {split: pooled_iou(data[split]["TP"].sum(axis=2),
                                 data[split]["FP"].sum(axis=2),
                                 data[split]["FN"].sum(axis=2))
                for split in SPLITS}  # (3, 5)
    for key, rec in meta.items():
        split, cfg, seed = key.split("|")
        rec["pooled_iou_full"] = float(full_iou[split][SEEDS.index(seed), CONFIGS.index(cfg)])

    # 两个 split 使用独立随机流：SeedSequence 派生两条子流
    rng_v = np.random.default_rng(np.random.SeedSequence([args.seed, 0]))
    rng_t = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    W = {"valid": bootstrap_weights(rng_v, data["valid"]["n"], args.reps),
         "test": bootstrap_weights(rng_t, data["test"]["n"], args.reps)}

    # 重抽样后的 pooled IoU：(reps, 3 seed, 5 config)
    boot_iou = {}
    invalid = {}
    for split in SPLITS:
        n = data[split]["n"]
        tp_mat = data[split]["TP"].reshape(3 * 5, n).T.astype(np.float64)
        fp_mat = data[split]["FP"].reshape(3 * 5, n).T.astype(np.float64)
        fn_mat = data[split]["FN"].reshape(3 * 5, n).T.astype(np.float64)
        wf = W[split].astype(np.float64)
        tp = wf @ tp_mat
        fp = wf @ fp_mat
        fn = wf @ fn_mat
        denom = tp + fp + fn
        iou = np.full_like(tp, np.nan)
        np.divide(tp, denom, out=iou, where=denom > 0)
        invalid[split] = int(np.sum(denom == 0))
        boot_iou[split] = iou.reshape(args.reps, 3, 5)

    # 逐 replicate 计算对比：同 seed 模型差 → 三 seed 均值
    contrast_boot = {}
    for cid, left, right, role in CONTRASTS:
        li, ri = CONFIGS.index(left), CONFIGS.index(right)
        dv = (boot_iou["valid"][:, :, li] - boot_iou["valid"][:, :, ri]).mean(axis=1)
        dt = (boot_iou["test"][:, :, li] - boot_iou["test"][:, :, ri]).mean(axis=1)
        contrast_boot[cid] = {"delta_valid": dv, "delta_test": dt, "D": dt - dv}

    # 原始点估计（全量数据，非重抽样）
    summary_rows = []
    for cid, left, right, role in CONTRASTS:
        li, ri = CONFIGS.index(left), CONFIGS.index(right)
        pv = float(np.mean(full_iou["valid"][:, li] - full_iou["valid"][:, ri]))
        pt = float(np.mean(full_iou["test"][:, li] - full_iou["test"][:, ri]))
        pD = pt - pv
        cb = contrast_boot[cid]
        sv, st, sD = summarize(cb["delta_valid"], "delta_valid"), \
            summarize(cb["delta_test"], "delta_test"), summarize(cb["D"], "D")
        summary_rows.append({
            "contrast_id": cid,
            "contrast": f"{left} - {right}",
            "role": role,
            "delta_valid_point": pv,
            "delta_valid_lo": sv["lo"], "delta_valid_hi": sv["hi"],
            "delta_test_point": pt,
            "delta_test_lo": st["lo"], "delta_test_hi": st["hi"],
            "D_point": pD, "D_lo": sD["lo"], "D_hi": sD["hi"],
            "n_reps": args.reps,
            "valid_reps_valid_split": args.reps - invalid["valid"],
            "valid_reps_test_split": args.reps - invalid["test"],
            "invalid_reps": invalid["valid"] + invalid["test"],
            "invalid_reason": "none" if (invalid["valid"] + invalid["test"]) == 0
                              else "denominator (TP+FP+FN)==0; replicate kept as NaN",
            "boot_mean_delta_valid": float(np.mean(cb["delta_valid"])),
            "boot_mean_delta_test": float(np.mean(cb["delta_test"])),
            "boot_mean_D": float(np.mean(cb["D"])),
            "sd_delta_valid_reps": sv["sd_reps"],
            "sd_delta_test_reps": st["sd_reps"],
            "sd_D_reps": sD["sd_reps"],
            "seeds_delta_valid_positive": int(np.sum((full_iou["valid"][:, li] - full_iou["valid"][:, ri]) > 0)),
            "seeds_delta_test_positive": int(np.sum((full_iou["test"][:, li] - full_iou["test"][:, ri]) > 0)),
            "seeds_D_positive": int(sum(1 for s in range(3)
                                        if (full_iou["test"][s, li] - full_iou["test"][s, ri])
                                        - (full_iou["valid"][s, li] - full_iou["valid"][s, ri]) > 0)),
        })

    # 逐种子效果
    per_seed_rows = []
    for cid, left, right, role in CONTRASTS:
        li, ri = CONFIGS.index(left), CONFIGS.index(right)
        vals = []
        for si, seed in enumerate(SEEDS):
            dv = float(full_iou["valid"][si, li] - full_iou["valid"][si, ri])
            dt = float(full_iou["test"][si, li] - full_iou["test"][si, ri])
            vals.append(dv)
            per_seed_rows.append({
                "contrast_id": cid, "contrast": f"{left} - {right}", "seed": seed,
                "iou_left_valid": float(full_iou["valid"][si, li]),
                "iou_right_valid": float(full_iou["valid"][si, ri]),
                "delta_valid": dv,
                "iou_left_test": float(full_iou["test"][si, li]),
                "iou_right_test": float(full_iou["test"][si, ri]),
                "delta_test": dt,
                "D": dt - dv,
            })
        # 三个种子的配对差 SD（ddof=1，仅描述性）
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        for row in per_seed_rows[-3:]:
            row["delta_valid_seed_sd"] = sd

    # 写出 CSV
    def write_csv(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(args.out_dir, "bootstrap_summary.csv"), summary_rows,
              list(summary_rows[0].keys()))
    write_csv(os.path.join(args.out_dir, "per_seed_effects.csv"), per_seed_rows,
              list(per_seed_rows[0].keys()))

    # 输入清单
    input_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "final zero-training paired patch bootstrap",
        "bootstrap_definition": {
            "reps": args.reps,
            "base_seed": args.seed,
            "rng_streams": {"valid": "SeedSequence([seed, 0])", "test": "SeedSequence([seed, 1])"},
            "resample_unit": "patch (with replacement)",
            "index_sharing": "same split & replicate shared by all configs and seeds",
            "statistic": "sum TP/FP/FN across resampled patches -> pooled IoU -> same-seed difference -> mean over 3 seeds",
            "interval": "percentile 2.5/97.5 (numpy.percentile, linear)",
            "pooled_iou_formula": "TP/(TP+FP+FN)",
            "invalid_rule": "denominator==0 -> NaN; reported, not redrawn",
        },
        "configs": CONFIGS,
        "seeds": SEEDS,
        "contrasts": [{"id": c, "left": l, "right": r, "role": ro} for c, l, r, ro in CONTRASTS],
        "tables": meta,
        "source_manifests": {},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "split_sizes": {s: data[s]["n"] for s in SPLITS},
        "label_consistency": "checked: positive_pixels/valid_pixels identical across all configs/seeds within split",
        "bootstrap_invalid_reps": invalid,
        "note": "Test is a historical benchmark already reused during research feedback; intervals do not restore independence or correct adaptive reuse/selection.",
    }
    man_root = os.path.join(args.input_root, "_manifests", "manifest_index.json")
    if os.path.exists(man_root):
        input_manifest["source_manifests"] = json.load(open(man_root, encoding="utf-8"))
    with open(os.path.join(args.out_dir, "input_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(input_manifest, f, indent=2, ensure_ascii=False)

    # 控制台摘要
    print(json.dumps({"summary": summary_rows, "invalid": invalid,
                      "split_sizes": {s: data[s]["n"] for s in SPLITS}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

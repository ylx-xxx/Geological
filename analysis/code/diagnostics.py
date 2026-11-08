# -*- coding: utf-8 -*-
"""补充诊断：bootstrap 均值与点估计的一致性、分位数稳定性、符号比例。"""
import csv, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap_analysis as ba

inp, out = sys.argv[1], sys.argv[2]
res = {}
for split in ba.SPLITS:
    order, TP, FP, FN, _ = ba.load_split(inp, split)
    res[split] = (TP, FP, FN, len(order))

# 复跑同一随机流，仅用于诊断（与主分析定义完全一致）
rngs = {"valid": np.random.default_rng(np.random.SeedSequence([ba.BASE_SEED, 0])),
        "test": np.random.default_rng(np.random.SeedSequence([ba.BASE_SEED, 1]))}
W = {s: ba.bootstrap_weights(rngs[s], res[s][3], ba.B_REPS) for s in ba.SPLITS}
full = {}
boot = {}
for split in ba.SPLITS:
    TP, FP, FN, n = res[split]
    full[split] = ba.pooled_iou(TP.sum(2), FP.sum(2), FN.sum(2))
    tp = W[split].astype(float) @ TP.reshape(15, n).T
    fp = W[split].astype(float) @ FP.reshape(15, n).T
    fn = W[split].astype(float) @ FN.reshape(15, n).T
    boot[split] = ba.pooled_iou(tp, fp, fn).reshape(ba.B_REPS, 3, 5)

diag = {}
for cid, left, right, _ in ba.CONTRASTS:
    li, ri = ba.CONFIGS.index(left), ba.CONFIGS.index(right)
    dv = (boot["valid"][:, :, li] - boot["valid"][:, :, ri]).mean(1)
    dt = (boot["test"][:, :, li] - boot["test"][:, :, ri]).mean(1)
    D = dt - dv
    pv = float(np.mean(full["valid"][:, li] - full["valid"][:, ri]))
    pt = float(np.mean(full["test"][:, li] - full["test"][:, ri]))
    diag[cid] = {
        "point_delta_valid": pv, "boot_mean_delta_valid": float(dv.mean()),
        "point_delta_test": pt, "boot_mean_delta_test": float(dt.mean()),
        "point_D": pt - pv, "boot_mean_D": float(D.mean()),
        "frac_reps_delta_valid_gt0": float((dv > 0).mean()),
        "frac_reps_delta_test_gt0": float((dt > 0).mean()),
        "frac_reps_D_gt0": float((D > 0).mean()),
        "frac_reps_opposite_sign_vt": float(((dv > 0) != (dt > 0)).mean()),
    }
json.dump(diag, open(os.path.join(out, "diagnostics.json"), "w"), indent=2, ensure_ascii=False)
for k, v in diag.items():
    print(k, {kk: round(vv, 6) for kk, vv in v.items()})

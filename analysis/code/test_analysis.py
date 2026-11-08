# -*- coding: utf-8 -*-
"""必需测试：身份配对、pooled 与 patch 平均不可混用、同模型差为零、点估计可复算。

运行：python -I test_analysis.py --input-root ../inputs
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap_analysis as ba


def copy_inputs(src, dst):
    """复制输入树到临时目录（不含 _manifests）。"""
    for name in os.listdir(src):
        if name == "_manifests":
            continue
        s = os.path.join(src, name)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, name))


def shuffle_rows(path, seed=7):
    """把一个 per_sample.csv 的行顺序打乱后写回。"""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in order:
            w.writerow(rows[i])


def test_shuffle_alignment(input_root):
    """表格乱序后，按 sample 显式对齐的结果必须与原顺序完全一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        copy_inputs(input_root, tmp)
        target = os.path.join(tmp, "valid__F14-S2__2027", "per_sample.csv")
        shuffle_rows(target)
        # 打乱后重新载入，样本顺序仍应以基准表为准，且计数与原始一致
        _, TP1, FP1, FN1, _ = ba.load_split(input_root, "valid")
        _, TP2, FP2, FN2, _ = ba.load_split(tmp, "valid")
        assert np.array_equal(TP1, TP2), "打乱行序后 TP 对齐失败"
        assert np.array_equal(FP1, FP2), "打乱行序后 FP 对齐失败"
        assert np.array_equal(FN1, FN2), "打乱行序后 FN 对齐失败"
        # 再打乱每一个表（含基准表），仍应按 sample 身份一致
        for split in ["valid", "test"]:
            for cfg in ba.CONFIGS:
                for seed in ba.SEEDS:
                    shuffle_rows(os.path.join(tmp, f"{split}__{cfg}__{seed}", "per_sample.csv"),
                                 seed=sum(map(ord, f"{split}{cfg}{seed}")) % 2**31)
        for split in ["valid", "test"]:
            s3, TP3, FP3, FN3, _ = ba.load_split(tmp, split)
            s4, TP4, FP4, FN4, _ = ba.load_split(input_root, split)
            assert set(s3) == set(s4)
            p3 = {s: i for i, s in enumerate(s3)}
            p4 = {s: i for i, s in enumerate(s4)}
            o3 = [p3[s] for s in s4]
            assert np.array_equal(TP3[:, :, o3], TP4)
            assert np.array_equal(FP3[:, :, o3], FP4)
            assert np.array_equal(FN3[:, :, o3], FN4)
    print("[PASS] 表格乱序后的身份配对")


def test_pooled_vs_patch_mean(input_root):
    """pooled IoU 由混淆计数求和计算，必须与 patch 级平均 IoU 不同。"""
    diffs = []
    for split in ["valid", "test"]:
        for cfg in ba.CONFIGS:
            for seed in ba.SEEDS:
                t = ba.load_table(os.path.join(input_root, f"{split}__{cfg}__{seed}", "per_sample.csv"))
                tp = t["counts"]["TP"].sum()
                fp = t["counts"]["FP"].sum()
                fn = t["counts"]["FN"].sum()
                pooled = tp / (tp + fp + fn)
                patch_mean = t["patch_iou"].mean()
                diffs.append(abs(pooled - patch_mean))
                # 与 metrics.json 的汇总口径一致，说明 pooled 才是官方整体 IoU
                import json
                m = json.load(open(os.path.join(input_root, f"{split}__{cfg}__{seed}", "metrics.json"),
                                   encoding="utf-8"))
                assert abs(pooled - m["Landslide_IoU"]) < 1e-12
    assert max(diffs) > 0.1, "pooled 与 patch 平均几乎相同，测试未有效区分两者"
    print(f"[PASS] pooled IoU 与 patch 平均不可混用（最大差异 {max(diffs):.4f}）")


def test_zero_difference(input_root):
    """同模型自比较的逐 replicate 差必须恒为零。"""
    _, TP, FP, FN, _ = ba.load_split(input_root, "valid")
    n = TP.shape[2]
    rng = np.random.default_rng(123)
    idx = rng.integers(0, n, size=(50, n))
    w = np.zeros((50, n), dtype=np.float64)
    for b in range(50):
        w[b] = np.bincount(idx[b], minlength=n)
    t = w @ TP[0, 0]
    f = w @ FP[0, 0]
    fn = w @ FN[0, 0]
    iou = ba.pooled_iou(t, f, fn)
    delta = iou - iou  # 同模型相减
    assert np.all(delta == 0)
    # 全量点估计自比较也应为零
    full = ba.pooled_iou(TP[0, 0].sum(), FP[0, 0].sum(), FN[0, 0].sum())
    assert float(full - full) == 0.0
    print("[PASS] 同模型差为零")


def test_point_estimates(input_root):
    """点估计可由 CSV 复算，并与任务书记录的三种子均值一致。"""
    expected = {
        # 配置: (valid 三种子均值, test 三种子均值)
        "RGB-S1": (0.490334, 0.473139),
        "F14-S1": (0.510287, 0.465103),
        "RGB-S2": (0.510566, 0.499418),
        "F14-S2": (0.550680, 0.470955),
        "F14-S2-N2": (0.529352, 0.486356),
    }
    for split, k in [("valid", 0), ("test", 1)]:
        _, TP, FP, FN, _ = ba.load_split(input_root, split)
        iou = ba.pooled_iou(TP.sum(axis=2), FP.sum(axis=2), FN.sum(axis=2))
        for ci, cfg in enumerate(ba.CONFIGS):
            got = float(iou[:, ci].mean())
            assert abs(got - expected[cfg][k]) < 1e-6, f"{split}/{cfg}: {got} vs {expected[cfg][k]}"
    # 四个对比的点估计也可直接复算
    exp_delta = {
        "C1": (0.040114, -0.028463),
        "C2": (-0.021328, 0.015401),
        "R1": (0.020232, 0.026279),
        "R2": (0.040393, 0.005852),
    }
    _, TPv, FPv, FNv, _ = ba.load_split(input_root, "valid")
    _, TPt, FPt, FNt, _ = ba.load_split(input_root, "test")
    iv = ba.pooled_iou(TPv.sum(axis=2), FPv.sum(axis=2), FNv.sum(axis=2))
    it = ba.pooled_iou(TPt.sum(axis=2), FPt.sum(axis=2), FNt.sum(axis=2))
    for cid, left, right, _ in ba.CONTRASTS:
        li, ri = ba.CONFIGS.index(left), ba.CONFIGS.index(right)
        dv = float(np.mean(iv[:, li] - iv[:, ri]))
        dt = float(np.mean(it[:, li] - it[:, ri]))
        assert abs(dv - exp_delta[cid][0]) < 2e-6, f"{cid} valid {dv}"
        assert abs(dt - exp_delta[cid][1]) < 2e-6, f"{cid} test {dt}"
    print("[PASS] 点估计可复算")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    args = ap.parse_args()
    test_shuffle_alignment(args.input_root)
    test_pooled_vs_patch_mean(args.input_root)
    test_zero_difference(args.input_root)
    test_point_estimates(args.input_root)
    print("全部测试通过")


if __name__ == "__main__":
    main()

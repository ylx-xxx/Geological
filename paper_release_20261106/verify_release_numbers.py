"""只读复核发布版主表与统计口径（不训练、不推理、不重跑 bootstrap）。

用法（任意目录，需在仓库根或本目录下执行）：
    python -I -X utf8 verify_release_numbers.py

核对内容：
1. 每个 run 的 metrics.json 中 Landslide_IoU 与由 TP/FP/FN 复算的值一致；
2. 由 per_sample.csv 汇总的 TP/FP/FN 与 metrics.json 的混淆矩阵一致；
3. 五配置 × 两种划分的三 seed 均值/样本标准差与 README 主表一致（四位小数）；
4. bootstrap_summary.csv 的点估计等于 per_seed_effects.csv 中同对照三 seed 差值的均值。
"""

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "results" / "inputs"

EXPECTED = {
    ("RGB-S1", "valid"): (0.4903, 0.0070),
    ("RGB-S1", "test"): (0.4731, 0.0141),
    ("F14-S1", "valid"): (0.5103, 0.0052),
    ("F14-S1", "test"): (0.4651, 0.0040),
    ("RGB-S2", "valid"): (0.5106, 0.0041),
    ("RGB-S2", "test"): (0.4994, 0.0105),
    ("F14-S2", "valid"): (0.5507, 0.0353),
    ("F14-S2", "test"): (0.4710, 0.0030),
    ("F14-S2-N2", "valid"): (0.5294, 0.0110),
    ("F14-S2-N2", "test"): (0.4864, 0.0147),
}

PREFIX = {
    "rgb_s1": "RGB-S1",
    "f14_s1": "F14-S1",
    "rgb_s2": "RGB-S2",
    "f14_s2": "F14-S2",
    "f14_s2_n2": "F14-S2-N2",
}


def run_name(path):
    """兼容 `valid__RGB-S1__2026` 与 `valid/rgb_s1/2026` 两种组织方式。"""
    parts = path.parts
    if len(parts) >= 1 and "__" in path.name:
        split, cfg, seed = path.name.split("__")
        return split, cfg, seed
    return path.parent.name, path.name, ""


def pooled_iou(tp, fp, fn):
    denom = tp + fp + fn
    return tp / denom if denom else 0.0


def main():
    failures = []

    # 1 + 2：逐 run 一致性
    run_means = {}
    for metrics_path in sorted(INPUTS.glob("**/metrics.json")):
        d = metrics_path.parent
        name = d.name
        if "__" in name:
            split, cfg, seed = name.split("__")
        else:
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        recalc = pooled_iou(m["TP"], m["FP"], m["FN"])
        if abs(recalc - m["Landslide_IoU"]) > 1e-9:
            failures.append(f"{name}: 混淆矩阵复算 IoU {recalc:.12f} != 记录 {m['Landslide_IoU']:.12f}")

        per_sample = d / "per_sample.csv"
        if per_sample.is_file():
            tp = fp = fn = 0
            with per_sample.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    tp += int(row["TP"]); fp += int(row["FP"]); fn += int(row["FN"])
            if (tp, fp, fn) != (m["TP"], m["FP"], m["FN"]):
                failures.append(f"{name}: 逐样本汇总 {tp}/{fp}/{fn} != metrics {m['TP']}/{m['FP']}/{m['FN']}")
        run_means.setdefault((cfg, split), []).append(m["Landslide_IoU"])

    # 3：五配置主表
    for (cfg, split), vals in sorted(run_means.items()):
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        exp = EXPECTED.get((cfg, split))
        if exp is None:
            failures.append(f"{cfg}/{split}: 不在 README 主表内")
            continue
        if round(mean, 4) != exp[0] or round(sd, 4) != exp[1]:
            failures.append(
                f"{cfg}/{split}: 复算 {mean:.4f}/{sd:.4f} != README {exp[0]:.4f}/{exp[1]:.4f}")
        print(f"{cfg:10s} {split:5s} n={len(vals)} mean={mean:.4f} sd={sd:.4f}")

    # 4：bootstrap 点估计 = 三 seed 差值均值
    summary = HERE / "results" / "bootstrap_summary.csv"
    effects = HERE / "results" / "per_seed_effects.csv"
    if summary.is_file() and effects.is_file():
        with effects.open(encoding="utf-8") as f:
            eff = list(csv.DictReader(f))
        by_contrast = {}
        for r in eff:
            by_contrast.setdefault(r["contrast_id"], []).append(r)
        with summary.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows = by_contrast.get(row["contrast_id"], [])
                if len(rows) != 3:
                    failures.append(f"{row['contrast_id']}: per_seed_effects 行数 {len(rows)} != 3")
                    continue
                for col, key in (("delta_valid", "delta_valid_point"),
                                 ("delta_test", "delta_test_point"),
                                 ("D", "D_point")):
                    calc = statistics.mean(float(r[col]) for r in rows)
                    if abs(calc - float(row[key])) > 1e-9:
                        failures.append(
                            f"{row['contrast_id']}.{key}: 复算 {calc:.12f} != 汇总 {float(row[key]):.12f}")
        print(f"对照点复核完成：{len(list(csv.DictReader(summary.open(encoding='utf-8'))))} 个对照")
    else:
        print("SKIP：未找到 bootstrap_summary.csv 或 per_seed_effects.csv")

    if failures:
        print("\n未通过项：")
        for f in failures:
            print(" -", f)
        return 1
    print("\n全部核对通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

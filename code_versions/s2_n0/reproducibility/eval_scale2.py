"""scale2 候选的唯一评价入口（新增文件，放入 reproducibility/）。

一次 TTA 推理同时得到：混淆矩阵与指标、前景概率 AP、逐样本表。
- --split 必须显式给出；split=test 还要显式 --confirm-test-once，避免开发阶段误读 TestData。
- 预测方式与 evaluate.predict 相同（三视角 logits 平均后 argmax），因此混淆计数与原评价口径一致。
- 记录实际执行的 scaled_model.py 等文件的 SHA-256。不做阈值搜索。
"""

import argparse
import csv
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_l4s_qz.model import SwinUPerNetL4S
from train_l4s_qz.scaled_model import SwinUPerNetL4SScaled
from .core import CHANNELS, NORMALIZATION, confusion, metrics, sha256, write_json
from .evaluate import EvaluationDataset, predict


def average_precision(prob, label):
    """与 sklearn.average_precision_score 同一定义（按不同阈值计算，含并列处理）。"""
    prob = np.asarray(prob, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    if label.sum() == 0:
        return float("nan")
    order = np.argsort(-prob, kind="stable")
    p, y = prob[order], label[order]
    distinct = np.where(np.diff(p))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = 1 + idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def build_eval_model(n_in, scale):
    if scale == 1:
        return SwinUPerNetL4S(in_chans=n_in, img_size=128)
    return SwinUPerNetL4SScaled(scale=scale, base_size=128, in_chans=n_in)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--confirm-test-once", action="store_true")
    ap.add_argument("--input-scale", required=True, type=int, choices=[1, 2])
    ap.add_argument("--channels", choices=list(CHANNELS), default="full14")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tta", type=int, choices=[0, 1], default=1)
    ap.add_argument("--amp", type=int, choices=[0, 1], default=1)
    args = ap.parse_args()
    if args.split == "test" and not args.confirm_test_once:
        raise ValueError("split=test 需要显式 --confirm-test-once（最终配置确定后只评一次）")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new output directory")
    actual = sha256(args.checkpoint)
    if actual != args.expected_sha256:
        raise ValueError("Checkpoint SHA256 differs from the frozen identity")

    ds = EvaluationDataset(args.data_root, args.split, args.channels)
    model = build_eval_model(len(CHANNELS[args.channels]), args.input_scale)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        Path(__file__).with_name("core.py"),
        Path(__file__).with_name("evaluate.py"),
        root / "train_l4s_qz/model.py",
        root / "train_l4s_qz/scaled_model.py",
    ]
    write_json(
        out / "protocol.json",
        {
            "checkpoint_sha256": actual,
            "split": args.split,
            "input_scale": args.input_scale,
            "channels": args.channels,
            "normalization": NORMALIZATION,
            "tta_views": 3 if args.tta else 1,
            "amp": bool(args.amp),
            "checkpoint_load": "strict",
            "sample_count": len(ds),
            "single_inference_pass": True,
            "versions": {
                n: importlib.metadata.version(n)
                for n in ["torch", "timm", "numpy", "h5py"]
            },
            "source_sha256": {str(p.relative_to(root)): sha256(p) for p in files},
        },
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    rows, probs, labels = [], [], []
    total = np.zeros((2, 2), dtype=np.int64)
    started = time.monotonic()
    with torch.inference_mode():
        for x, y, indices in loader:
            x = x.to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=bool(args.amp) and device.type == "cuda",
            ):
                logits = predict(model, x, bool(args.tta)).float()
            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
            fg = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            for p, g, f, i in zip(pred, y.numpy(), fg, indices.tolist()):
                cm = confusion(p, g)
                total += cm
                valid = g != 255
                probs.append(f[valid].astype(np.float32))
                labels.append(g[valid] == 1)
                row = {
                    "sample": ds.samples[i][0].name,
                    "positive_pixels": int((g == 1).sum()),
                    "valid_pixels": int(valid.sum()),
                    **metrics(cm),
                }
                row["foreground_patch_iou"] = (
                    row["Landslide_IoU"] if row["positive_pixels"] else ""
                )
                rows.append(row)
    ap_value = average_precision(np.concatenate(probs), np.concatenate(labels))
    summed = np.array(
        [
            [sum(r["TN"] for r in rows), sum(r["FP"] for r in rows)],
            [sum(r["FN"] for r in rows), sum(r["TP"] for r in rows)],
        ]
    )
    if not np.array_equal(summed, total):
        raise AssertionError("Sample/global counts differ")
    with (out / "per_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        **metrics(total),
        "Landslide_AP": ap_value,
        "samples": len(rows),
        "confusion_matrix": total.tolist(),
        "seconds": time.monotonic() - started,
    }
    write_json(out / "metrics.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

"""Audited training: validation selection or a prespecified fixed-epoch TrainVal refit."""

import argparse
import csv
import hashlib
import random
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from train_l4s_qz.model import (
    RGB_TARGET_CHANNELS,
    SwinUPerNetL4S,
    adapt_patch_embed_weight,
)
from train_l4s_qz.stca import (
    L4S_EXTRA_CHANNELS,
    L4S_RGB_CHANNELS,
    STCA,
    STCA_EMBED_DIM,
    STCAPatchEmbed,
    adapter_parameter_count,
)
from train_l4s_qz.losses import CombinedLoss
from train_l4s_qz.scaled_model import build_swin
from train_l4s_qz.time_budget import time_limit_reached
from .core import (
    CHANNELS,
    NORMALIZATION,
    NORMALIZATION_N2,
    apply_normalization,
    load_terrain_stats,
    pairs,
    read_h5,
    sha256,
    confusion,
    metrics,
    write_json,
    verify_selection_split,
)
from .evaluate import predict


class TrainingDataset(Dataset):
    """Cache normalized input; augmentation follows the historical ablation recipe."""

    def __init__(self, root, split, channels, norm_mode="n0", terrain_stats=None):
        if split not in ["train", "val"]:
            raise ValueError("Training cannot access test data")
        self.split = split
        self.norm_mode = norm_mode
        self.samples = pairs(root, split)
        self.cache = []
        for ip, mp in self.samples:
            x = apply_normalization(read_h5(ip, "img"), norm_mode, terrain_stats)[
                CHANNELS[channels]
            ]
            y = np.squeeze(read_h5(mp, "mask")).astype(np.int64)
            if not np.isin(y, [0, 1, 255]).all():
                raise ValueError(f"Invalid labels: {mp}")
            if x.shape[-2:] != (128, 128) or y.shape != (128, 128):
                raise ValueError("Expected native 128 x 128 patches")
            self.cache.append((x, y))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        x, y = self.cache[i]
        if self.split == "train":
            if random.random() < 0.5:
                x = x[:, :, ::-1]
                y = y[:, ::-1]
            if random.random() < 0.5:
                x = x[:, ::-1, :]
                y = y[::-1, :]
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                x = np.rot90(x, k, (1, 2))
                y = np.rot90(y, k)
            if random.random() < 0.3:
                x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def resize_bias(source, target):
    """Bicubic interpolation of a square relative-position grid, per head."""
    old = int(source.shape[0] ** 0.5)
    new = int(target.shape[0] ** 0.5)
    if (
        source.ndim != 2
        or target.ndim != 2
        or old * old != source.shape[0]
        or new * new != target.shape[0]
        or source.shape[1] != target.shape[1]
    ):
        raise ValueError("Unsupported relative-position bias shape")
    grid = source.T.reshape(1, source.shape[1], old, old).float()
    grid = torch.nn.functional.interpolate(
        grid, size=(new, new), mode="bicubic", align_corners=False
    )
    return grid.reshape(source.shape[1], new * new).T.to(dtype=target.dtype)


def patch_transfer(source, target, extra="mean", target_channels=None):
    if extra not in {"mean", "random", "zero"}:
        raise ValueError("Unknown extra-channel initialization")
    converted = adapt_patch_embed_weight(
        source, target, extra=extra, target_channels=target_channels
    )
    if converted is None:
        raise ValueError("Unsupported patch embedding shape")
    return converted


def transfer(model, path, extra="mean", target_channels=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    target = model.state_dict()
    converted = {}
    adapted = []
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key not in target:
            continue
        if value.shape == target[key].shape:
            converted[key] = value
        elif key.endswith("patch_embed.proj.weight"):
            converted[key] = patch_transfer(value, target[key], extra, target_channels)
            adapted.append(key)
        elif key.endswith("relative_position_bias_table"):
            converted[key] = resize_bias(value, target[key])
            adapted.append(key)
    allowed = {"fuse.2.weight", "fuse.2.bias"}
    missing = set(target) - set(converted)
    # 适配器参数在源 checkpoint 中不存在，属于预期的随机初始化，不算漏载。
    adapter_random = sorted(k for k in missing if ".adapter." in k)
    if missing - allowed - set(adapter_random):
        raise RuntimeError(
            f"Incomplete LoveDA transfer: {sorted(missing - allowed - set(adapter_random))}"
        )
    model.load_state_dict(converted, strict=False)
    return {
        "direct_or_adapted_keys": len(converted),
        "adapted_keys": adapted,
        "new_classifier_keys": sorted(missing - set(adapter_random)),
        "adapter_random_keys": adapter_random,
        "source_sha256": sha256(path),
    }


def state_sha256(model):
    """对模型当前全部参数做确定性指纹，用于记录初始化状态。"""
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


ENCODER_PREFIX = "encoder."


def _convert_imagenet_key(key):
    """把 timm 原始 ImageNet 键转换成 features_only encoder 的键名。"""
    key = key.removeprefix("module.").removeprefix("model.")
    if key.startswith("head.") or key.startswith("norm."):
        return None
    for i in range(4):
        key = key.replace(f"layers.{i}.", f"layers_{i}.")
    return ENCODER_PREFIX + key


def _read_pretrained(path):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def encoder_only_transfer(model, path, extra="mean", imagenet=False, target_channels=None):
    """只加载 encoder 权重，解码器保持随机初始化。

    imagenet=True 时按 timm 原始键名转换；False 时从 LoveDA 完整模型中
    仅抽取 encoder.* 键。两条路径都要求 encoder 全部命中，避免静默漏载。
    """
    state = _read_pretrained(path)
    target = model.state_dict()
    converted, adapted = {}, []
    for key, value in state.items():
        if imagenet:
            key = _convert_imagenet_key(key)
            if key is None:
                continue
        else:
            key = key.removeprefix("module.")
            # 只接受 encoder 键，避免把源模型的解码器一并载入
            if not key.startswith(ENCODER_PREFIX):
                continue
        if key not in target:
            continue
        if value.shape == target[key].shape:
            converted[key] = value
        elif key.endswith("patch_embed.proj.weight"):
            converted[key] = patch_transfer(value, target[key], extra, target_channels)
            adapted.append(key)
        elif key.endswith("relative_position_bias_table"):
            converted[key] = resize_bias(value, target[key])
            adapted.append(key)
    encoder_target = {k for k in target if k.startswith(ENCODER_PREFIX)}
    adapter_random = sorted(k for k in encoder_target if ".adapter." in k)
    missing_encoder = sorted(encoder_target - set(converted) - set(adapter_random))
    if missing_encoder:
        raise RuntimeError(f"Incomplete encoder transfer: {missing_encoder}")
    decoder_random = sorted(k for k in target if not k.startswith(ENCODER_PREFIX))
    model.load_state_dict(converted, strict=False)
    return {
        "direct_or_adapted_keys": len(converted),
        "adapted_keys": adapted,
        "encoder_keys_loaded": len(converted),
        "random_decoder_keys": decoder_random,
        "random_decoder_key_count": len(decoder_random),
        "adapter_random_keys": adapter_random,
        "source_sha256": sha256(path),
    }


def expand_patch_embed(model, in_chans):
    """把 3 通道 patch_embed 扩展为多通道，权重先置零，随后由迁移填充。

    在完整模型构造之后执行，因此不会改变其他模块的随机数消耗顺序。
    """
    proj = model.encoder.patch_embed.proj
    new = torch.nn.Conv2d(
        in_chans,
        proj.out_channels,
        proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=proj.bias is not None,
        device=proj.weight.device,
        dtype=proj.weight.dtype,
    )
    with torch.no_grad():
        new.weight.zero_()
        if proj.bias is not None:
            new.bias.copy_(proj.bias)
    model.encoder.patch_embed.proj = new


def build_l4s_model(in_chans, patch_channel_map="first3", img_size=128, scale=1):
    """训练与检查共用的模型构造路径。

    rgb 映射且输入为多通道时，先用 3 通道构造（与 RGB 基线消耗完全相同的随机数），
    再替换 patch_embed 的输入通道。这样同种子下解码器等共享参数的初始化与 RGB 基线一致，
    不需要也不允许从已训练的基线权重复制参数。
    """
    if patch_channel_map == "rgb" and in_chans > 3:
        model = build_swin(3, img_size, scale)
        expand_patch_embed(model, in_chans)
        return model
    return build_swin(in_chans, img_size, scale)


def build_stca_model(in_chans=14, img_size=128, fusion="mul", adapter_seed=None):
    """STCA 构造路径：共享 RGB 主干完整构造后，用独立同种子 RNG 构建适配器。

    - 主干仍是 3 通道构造，patch_embed 保持预训练形状，不扩展、不复制权重；
    - 适配器在保存/恢复 CPU 与 CUDA RNG 状态的上下文中构造，因此
      共享参数（含 decoder）与 RGB 基线逐位一致，训练期 dropout 的随机起点也不被改变；
    - M/A 两臂传入同一 adapter_seed，支路初始参数逐个相同。
    """
    if in_chans != 3 + len(L4S_EXTRA_CHANNELS):
        raise ValueError("STCA 只接受 full14（3 个物理 RGB + 11 个额外通道）输入")
    model = SwinUPerNetL4S(in_chans=3, img_size=img_size)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if adapter_seed is not None:
            torch.manual_seed(adapter_seed)
        adapter = STCA(
            in_chans=len(L4S_EXTRA_CHANNELS),
            embed_dim=STCA_EMBED_DIM,
            fusion=fusion,
        )
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    model.encoder.patch_embed = STCAPatchEmbed(model.encoder.patch_embed, adapter)
    return model


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loveda-checkpoint", default="")
    ap.add_argument("--imagenet-checkpoint", default="")
    ap.add_argument(
        "--initialization",
        choices=["loveda", "loveda_encoder", "imagenet", "random"],
        default="loveda",
    )
    ap.add_argument("--patch-extra", choices=["mean", "random", "zero"], default="mean")
    # 多通道初始化映射：first3 为历史原位复制（I-03 错配），rgb 为物理 RGB→[3,2,1]。
    ap.add_argument(
        "--patch-channel-map",
        choices=["first3", "rgb"],
        default="first3",
        help="first3: 源 RGB 核原位复制到前 3 个通道（历史行为）；rgb: 映射到物理 RGB 位置",
    )
    ap.add_argument(
        "--protocol", choices=["train_only", "trainval_fixed"], default="train_only"
    )
    ap.add_argument("--channels", choices=list(CHANNELS), required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=4)
    # 0 表示不设时间上限；只有正数才触发 epoch 内/epoch 间的计时退出。
    ap.add_argument("--max-seconds", type=int, default=0)
    # 本轮唯一新增训练目标开关：空标注 patch 前景概率惩罚。
    # 默认 0.0 时必须与历史 CombinedLoss 完全一致（见单元测试）。
    ap.add_argument("--empty-penalty-weight", type=float, default=0.0)
    # STCA 适配器：默认关闭，保持历史构造与前向路径不变。
    ap.add_argument("--adapter", choices=["none", "stca"], default="none")
    ap.add_argument("--adapter-fusion", choices=["mul", "add"], default="mul")
    # 目标端输入放大倍数：1 为原流程；2 为 128→256（仅模型内部放大，数据/标签/评价不变）。
    ap.add_argument("--input-scale", type=int, choices=[1, 2], default=1)
    # 归一化模式：n0 为历史逐 patch 标准化；n2 仅替换 slope/DEM 为固定 Train 统计。
    ap.add_argument("--norm-mode", choices=["n0", "n2"], default="n0")
    ap.add_argument("--terrain-stats", default="")
    args = ap.parse_args()
    if args.input_scale != 1 and args.adapter != "none":
        raise ValueError("--input-scale 只能与 --adapter none 组合")
    if args.norm_mode == "n2":
        if not args.terrain_stats:
            raise ValueError("n2 requires --terrain-stats")
        terrain_stats = load_terrain_stats(args.terrain_stats)
    else:
        if args.terrain_stats:
            raise ValueError("n0 不接受 --terrain-stats，避免误配")
        terrain_stats = None
    if args.initialization in {"loveda", "loveda_encoder"} and not args.loveda_checkpoint:
        raise ValueError("This initialization requires --loveda-checkpoint")
    if args.initialization == "imagenet" and not args.imagenet_checkpoint:
        raise ValueError("imagenet initialization requires --imagenet-checkpoint")
    n_input = len(CHANNELS[args.channels])
    if args.patch_channel_map == "rgb":
        if n_input < 4:
            raise ValueError("rgb 映射只适用于多通道输入（当前通道数 < 4）")
        if args.initialization == "random":
            raise ValueError("random 初始化没有源 RGB 核，不能使用 rgb 映射")
    if args.adapter == "stca":
        if args.channels != "full14":
            raise ValueError("STCA 只支持 --channels full14（3 个物理 RGB + 11 个额外通道）")
        if args.patch_channel_map != "rgb":
            raise ValueError("STCA 需要 --patch-channel-map rgb")
    patch_target_channels = (
        None if args.patch_channel_map == "first3" else RGB_TARGET_CHANNELS
    )
    # STCA 不扩通道：patch_embed 仍是 3 通道，迁移时形状直接相等，无需通道映射。
    transfer_target_channels = None if args.adapter == "stca" else patch_target_channels
    if args.protocol == "train_only":
        verify_selection_split("val")
    if args.epochs < 1 or args.batch_size < 2 or args.max_seconds < 0:
        raise ValueError("Invalid training limits")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new run directory")
    if "checkpoints" in out.parts:
        raise ValueError(
            "Use a new runs directory; original checkpoint directories are protected"
        )
    out.mkdir(parents=True)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    source_root = Path(__file__).resolve().parents[1]
    source_files = list(Path(__file__).parent.glob("*.py")) + [
        source_root / "train_l4s_qz/model.py",
        source_root / "train_l4s_qz/losses.py",
        source_root / "train_l4s_qz/stca.py",
        source_root / "train_l4s_qz/scaled_model.py",
    ]
    patch_init = {
        "loveda": "positional_source_" + args.patch_extra + "_extra",
        "loveda_encoder": "encoder_only_" + args.patch_extra + "_extra",
        "imagenet": "imagenet_encoder_" + args.patch_extra + "_extra",
        "random": "all_random",
    }[args.initialization]
    if patch_target_channels is not None:
        patch_init += "_rgbmap"
    if args.adapter == "stca":
        patch_init += "_stca_3ch_embed"
    config = vars(args) | {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "selection_split": "val" if args.protocol == "train_only" else None,
        "selection_metric": (
            "Landslide_IoU"
            if args.protocol == "train_only"
            else "prespecified_final_epoch"
        ),
        "normalization": NORMALIZATION if args.norm_mode == "n0" else NORMALIZATION_N2,
        "normalization_mode": args.norm_mode,
        "terrain_stats_file": terrain_stats["path"] if terrain_stats else None,
        "terrain_stats_sha256": terrain_stats["sha256"] if terrain_stats else None,
        "terrain_stats_train_list_sha256": (
            terrain_stats["train_list_sha256"] if terrain_stats else None
        ),
        "terrain_stats_channels": terrain_stats["channels"] if terrain_stats else None,
        "terrain_stats_n_pixels": terrain_stats["n_pixels"] if terrain_stats else None,
        "channel_indices": CHANNELS[args.channels],
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": 0.05,
        "label_smoothing": 0.02,
        "loss": (
            "weighted_ce_plus_weighted_dice"
            if args.empty_penalty_weight == 0.0
            else "weighted_ce_plus_weighted_dice_plus_empty_fp"
        ),
        "empty_penalty_weight": args.empty_penalty_weight,
        "patch_init": patch_init,
        "patch_channel_map": args.patch_channel_map,
        "patch_target_channels": (
            list(patch_target_channels) if patch_target_channels is not None else None
        ),
        "input_channel_count": n_input,
        "model_build": (
            "rgb_first_then_expand"
            if patch_target_channels is not None and n_input > 3
            else "direct"
        ),
        "val_tta_views": 3,
        "initialization_source": {
            "loveda": args.loveda_checkpoint,
            "loveda_encoder": args.loveda_checkpoint,
            "imagenet": args.imagenet_checkpoint,
            "random": None,
        }[args.initialization],
        "source_checkpoint_sha256": (
            sha256(
                args.imagenet_checkpoint
                if args.initialization == "imagenet"
                else args.loveda_checkpoint
            )
            if args.initialization != "random"
            else None
        ),
        "code_sha256": {
            str(p.relative_to(source_root)): sha256(p) for p in source_files
        },
        "torch": torch.__version__,
        "cudnn_benchmark": False,
        "note": "new prospective run, not a replacement for historical scores",
    }
    if args.adapter == "stca":
        config["adapter"] = "stca_v1"
        config["adapter_cli"] = args.adapter
        config["adapter_fusion"] = args.adapter_fusion
        config["adapter_injection_point"] = "patch_embed_output_before_first_swin_block"
        config["adapter_input_split"] = {
            "rgb": list(L4S_RGB_CHANNELS),
            "extra": list(L4S_EXTRA_CHANNELS),
        }
        config["adapter_ln"] = {"eps": 1e-5, "elementwise_affine": True}
        config["adapter_init"] = "torch_default_init_with_zero_last_conv"
        config["adapter_seed"] = args.seed
        config["adapter_expected_params"] = 11616
        # STCA 不扩展 patch_embed 的输入通道，迁移时形状直接相等。
        config["patch_target_channels"] = None
        config["patch_extra_effective"] = None
        config["model_build"] = "stca_3ch_rgb_embed_plus_11ch_branch"
    else:
        config["adapter"] = None
    write_json(out / "config.json", config)
    train = TrainingDataset(
        args.data_root, "train", args.channels, args.norm_mode, terrain_stats
    )
    val = (
        TrainingDataset(
            args.data_root, "val", args.channels, args.norm_mode, terrain_stats
        )
        if args.protocol == "train_only"
        else None
    )
    if args.protocol == "trainval_fixed":
        additional = TrainingDataset(
            args.data_root, "val", args.channels, args.norm_mode, terrain_stats
        )
        train.samples.extend(additional.samples)
        train.cache.extend(additional.cache)
        del additional
    # Train 清单身份：排序后的相对影像路径，与阶段 0 使用同一约定。
    config["train_list_sha256"] = hashlib.sha256(
        "\n".join(
            Path(ip).resolve().relative_to(Path(args.data_root).resolve()).as_posix()
            for ip, _ in train.samples
        ).encode("utf-8")
    ).hexdigest()
    counts = np.array(
        [sum(int((y == c).sum()) for _, y in train.cache) for c in [0, 1]],
        dtype=np.float64,
    )
    weights = 1 / np.sqrt(counts / counts.sum() + 1e-12)
    weights = np.clip(weights / weights.mean(), 0.5, 2.5)
    config["class_weights"] = weights.tolist()
    config["train_samples"] = len(train)
    config["val_samples"] = len(val) if val else 0
    write_json(out / "config.json", config)
    # Save sample identity without reading TestData.
    for name, ds in [("train", train), ("val", val)]:
        if ds is None:
            continue
        write_json(
            out / f"{name}_manifest.json",
            [
                {
                    "sample": ip.name,
                    "source_image": str(ip.relative_to(Path(args.data_root))),
                    "image_sha256": sha256(ip),
                    "mask_sha256": sha256(mp),
                }
                for ip, mp in ds.samples
            ],
        )
    gen = torch.Generator().manual_seed(args.seed)
    loaders = [
        DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(ds is train),
            num_workers=args.workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=gen if ds is train else None,
            persistent_workers=args.workers > 0,
        )
        for ds in [train, val]
        if ds is not None
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.adapter == "stca":
        model = build_stca_model(
            len(CHANNELS[args.channels]),
            img_size=128,
            fusion=args.adapter_fusion,
            adapter_seed=args.seed,
        )
    else:
        model = build_l4s_model(
            len(CHANNELS[args.channels]),
            args.patch_channel_map,
            img_size=128,
            scale=args.input_scale,
        )
    if args.initialization == "loveda":
        transfer_info = transfer(
            model, args.loveda_checkpoint, args.patch_extra, transfer_target_channels
        )
    elif args.initialization == "loveda_encoder":
        transfer_info = encoder_only_transfer(
            model,
            args.loveda_checkpoint,
            args.patch_extra,
            imagenet=False,
            target_channels=transfer_target_channels,
        )
    elif args.initialization == "imagenet":
        transfer_info = encoder_only_transfer(
            model,
            args.imagenet_checkpoint,
            args.patch_extra,
            imagenet=True,
            target_channels=transfer_target_channels,
        )
    else:
        transfer_info = {
            "initialization": "all_random",
            "direct_or_adapted_keys": 0,
            "source_sha256": None,
            "note": (
                "timm create_model(pretrained=False)；未加载任何 checkpoint，"
                "encoder 与 decoder 均为随机初始化"
            ),
        }
    if args.adapter == "stca":
        adapter_params = adapter_parameter_count(model.encoder.patch_embed.adapter)
        if adapter_params != config["adapter_expected_params"]:
            raise RuntimeError(
                f"STCA 参数数量不符: {adapter_params} != {config['adapter_expected_params']}"
            )
        config["adapter_params"] = adapter_params
        transfer_info["adapter_params"] = adapter_params
    transfer_info["init_parity"] = getattr(model, "init_parity", None)
    transfer_info["initial_state_sha256"] = state_sha256(model)
    config["initial_state_sha256"] = transfer_info["initial_state_sha256"]
    config["initialization_description"] = transfer_info.get("note", args.initialization)
    write_json(out / "config.json", config)
    write_json(out / "transfer.json", transfer_info)
    model = model.to(device, memory_format=torch.channels_last)
    criterion = CombinedLoss(
        class_weights=weights.tolist(),
        label_smoothing=0.02,
        empty_penalty_weight=args.empty_penalty_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best = -1.0
    best_epoch = 0
    done = 0
    reason = "completed"
    for epoch in range(1, args.epochs + 1):
        if time_limit_reached(args.max_seconds, time.monotonic() - started):
            reason = "time_limit"
            break
        model.train()
        loss_sum = 0.0
        batches = 0
        timed_out = False
        for x, y in loaders[0]:
            if time_limit_reached(args.max_seconds, time.monotonic() - started):
                timed_out = True
                break
            x = x.to(device, memory_format=torch.channels_last)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            batches += 1
        if timed_out:
            reason = "time_limit_partial_epoch"
            break
        if args.protocol == "trainval_fixed":
            scheduler.step()
            done = epoch
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / max(batches, 1),
                "seconds": time.monotonic() - started,
            }
            with (out / "history.csv").open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if epoch == 1:
                    w.writeheader()
                w.writerow(row)
            if epoch == args.epochs:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "config": config,
                        "selection_split": None,
                        "selection_metric": "prespecified_final_epoch",
                    },
                    out / "fixed_final.pt",
                )
                best_epoch = epoch
            print(row, flush=True)
            continue
        model.eval()
        cm = np.zeros((2, 2), dtype=np.int64)
        with torch.inference_mode():
            for x, y in loaders[1]:
                x = x.to(device, memory_format=torch.channels_last)
                with torch.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    p = predict(model, x, True).argmax(1).cpu().numpy()
                cm += confusion(p, y.numpy())
        score = metrics(cm)
        scheduler.step()
        done = epoch
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            "val_landslide_iou": score["Landslide_IoU"],
            "val_landslide_f1": score["Landslide_F1"],
            "val_precision": score["Landslide_Precision"],
            "val_recall": score["Landslide_Recall"],
            "seconds": time.monotonic() - started,
        }
        with (out / "history.csv").open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 1:
                w.writeheader()
            w.writerow(row)
        if score["Landslide_IoU"] > best:
            best = score["Landslide_IoU"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "config": config,
                    "selection_split": "val",
                    "selection_metric": "Landslide_IoU",
                    "validation_metrics": score,
                },
                out / "best.pt",
            )
        print(row, flush=True)
    summary = {
        "completed_epochs": done,
        "planned_epochs": args.epochs,
        "status": reason,
        "best_validation_iou": best if val is not None else None,
        "best_epoch": best_epoch,
        "selection_split": "val" if val is not None else None,
        "seconds": time.monotonic() - started,
        "test_evaluated": False,
    }
    if (out / "best.pt").exists():
        summary["checkpoint_sha256"] = sha256(out / "best.pt")
    if (out / "fixed_final.pt").exists():
        summary["checkpoint_sha256"] = sha256(out / "fixed_final.pt")
    write_json(out / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

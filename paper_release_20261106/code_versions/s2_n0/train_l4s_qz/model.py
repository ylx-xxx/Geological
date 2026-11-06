import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SwinUPerNetL4S(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_chans=14,
        backbone="swin_tiny_patch4_window7_224.ms_in1k",
        img_size=128,
        fpn_dim=256,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            backbone,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
            in_chans=in_chans,
        )

        self.channels = self.encoder.feature_info.channels()
        print(f"[Info] L4S encoder channels: {self.channels}")

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1) for c in self.channels
        ])

        self.smooth = nn.ModuleList([
            ConvBNReLU(fpn_dim, fpn_dim) for _ in self.channels
        ])

        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(self.channels[-1], fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for s in [1, 2, 3, 6]
        ])

        self.ppm_bottleneck = ConvBNReLU(fpn_dim * 4 + self.channels[-1], fpn_dim)

        self.fuse = nn.Sequential(
            ConvBNReLU(fpn_dim * 4, fpn_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def _to_nchw(self, feats):
        out = []
        for i, (f, c) in enumerate(zip(feats, self.channels)):
            if f.ndim != 4:
                raise RuntimeError(f"Feature {i} should be 4D, got {tuple(f.shape)}")

            if f.shape[1] == c:
                out.append(f.contiguous())
            elif f.shape[-1] == c:
                out.append(f.permute(0, 3, 1, 2).contiguous())
            else:
                raise RuntimeError(
                    f"Cannot infer feature layout for feature {i}, "
                    f"expected C={c}, got shape={tuple(f.shape)}"
                )
        return out

    def forward(self, x):
        input_size = x.shape[-2:]

        feats = self.encoder(x)
        feats = self._to_nchw(feats)

        c1, c2, c3, c4 = feats

        ppm_outs = [c4]
        for ppm in self.ppm:
            y = ppm(c4)
            y = F.interpolate(y, size=c4.shape[-2:], mode="bilinear", align_corners=False)
            ppm_outs.append(y)

        p4 = self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))

        laterals = [
            self.lateral[0](c1),
            self.lateral[1](c2),
            self.lateral[2](c3),
            p4,
        ]

        for i in range(3, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        outs = [self.smooth[i](laterals[i]) for i in range(4)]

        target_size = outs[0].shape[-2:]
        outs = [
            F.interpolate(o, size=target_size, mode="bilinear", align_corners=False)
            for o in outs
        ]

        out = self.fuse(torch.cat(outs, dim=1))
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return out


# 源模型（LoveDA / ImageNet）的卷积核按标准 R,G,B 排列，对应本文件的位置 0,1,2。
RGB_SOURCE_CHANNELS = (0, 1, 2)
# L4S 存储顺序中物理 RGB 的位置：B4→R、B3→G、B2→B，即 [3, 2, 1]。
RGB_TARGET_CHANNELS = (3, 2, 1)


def adapt_patch_embed_weight(old_w, new_w, extra="mean", target_channels=None):
    """把 3 通道 RGB 卷积核迁移到多通道输入。

    old_w: [out, C_src, k, k]，源模型按 R,G,B 排列；
    new_w: [out, C_dst, k, k]，目标模型当前的随机初始化。

    target_channels:
        None   —— 历史行为：源核原位复制到目标前 C_src 个位置（即 B1/B2/B3）。
                  这就是 I-03 登记的错配模式，仅用于复现已登记的历史结果；
        (3,2,1) —— 物理 RGB 映射：R→B4、G→B3、B→B2，其余通道按 extra 初始化。

    extra: 其余通道的初始化方式
        "mean"   源核通道均值（历史默认）；
        "zero"   置零（本轮多通道基线的规则）；
        "random" 保留目标模型的随机初始化。
    """
    if old_w.ndim != 4 or new_w.ndim != 4:
        return None

    if old_w.shape[0] != new_w.shape[0] or old_w.shape[2:] != new_w.shape[2:]:
        return None

    if extra not in {"mean", "zero", "random"}:
        raise ValueError(f"未知的额外通道初始化方式: {extra}")

    out = new_w.clone()
    old_in = old_w.shape[1]
    new_in = new_w.shape[1]

    if target_channels is None:
        # 历史路径：逐行保持原实现，避免复现结果出现任何数值偏移。
        copy_ch = min(old_in, new_in)
        out[:, :copy_ch] = old_w[:, :copy_ch]
        if new_in > old_in and extra != "random":
            mean_w = old_w.mean(dim=1, keepdim=True)
            fill = 0 if extra == "zero" else mean_w
            out[:, old_in:new_in] = (
                fill.repeat(1, new_in - old_in, 1, 1) if extra == "mean" else fill
            )
        return out

    if old_in > new_in:
        raise ValueError("目标通道数少于源通道数，无法建立映射")

    positions = [int(i) for i in target_channels]
    if len(positions) != old_in or len(set(positions)) != old_in:
        raise ValueError("目标通道位置的数量必须等于源通道数且互不重复")
    if min(positions) < 0 or max(positions) >= new_in:
        raise ValueError(f"目标通道位置超出范围 [0, {new_in})")

    for src, dst in enumerate(positions):
        out[:, dst] = old_w[:, src]

    extras = [i for i in range(new_in) if i not in set(positions)]
    if extras:
        if extra == "mean":
            mean_w = old_w.mean(dim=1, keepdim=True)
            out[:, extras] = mean_w.repeat(1, len(extras), 1, 1)
        elif extra == "zero":
            out[:, extras] = 0
        # extra == "random" 时保留 new_w 的随机初始化
    return out


def load_loveda_pretrained(model, ckpt_path):
    """历史加载入口：保留原位复制行为（I-03 错配），仅用于复现已登记结果。

    新流程请使用 reproducibility/train.py 的 --patch-channel-map rgb。
    """
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[Warning] LoveDA checkpoint not found: {ckpt_path}")
        return

    print(f"[Info] Loading LoveDA checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model_state = model.state_dict()
    converted = {}

    loaded = 0
    adapted = 0
    skipped = 0

    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]

        if nk not in model_state:
            skipped += 1
            continue

        if model_state[nk].shape == v.shape:
            converted[nk] = v
            loaded += 1
            continue

        # Adapt 3-channel LoveDA patch embedding to 14-channel L4S patch embedding.
        if "patch_embed.proj.weight" in nk:
            new_v = adapt_patch_embed_weight(v, model_state[nk])
            if new_v is not None and new_v.shape == model_state[nk].shape:
                converted[nk] = new_v
                adapted += 1
                continue

        skipped += 1

    msg = model.load_state_dict(converted, strict=False)

    print("[Info] LoveDA pretrained transfer finished.")
    print(f"[Info] Direct loaded keys : {loaded}")
    print(f"[Info] Adapted keys       : {adapted}")
    print(f"[Info] Skipped keys       : {skipped}")
    print(f"[Info] Missing keys       : {len(msg.missing_keys)}")
    print(f"[Info] Unexpected keys    : {len(msg.unexpected_keys)}")

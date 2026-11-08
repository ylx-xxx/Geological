"""光谱—地形条件调制适配器（STCA v1）。

设计见 `mc_adapter_design_v1.md`，本文件只实现一个候选，不提供第二模块：

    e = AvgPool_{4x4,4}(x_extra)                       # 与 patch 4x4、步长 4 对齐
    u = GELU(LN(DWConv3x3(Conv1x1(e))))                 # 11 -> 96 -> 96
    s = ZeroConv1x1(u)                                 # 96 -> 96，权重与偏置全零
    M（乘性）: H = H0 * (1 + s)
    A（加性）: H = H0 + s

H0 为 timm Swin 原有 3 通道 patch embedding 的完整输出（含其原有 normalization）。
注入点在 patch_embed 之后、首个 Swin block 之前，只注入一次。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 固定输入语义，不搜索通道子集：
# 物理 RGB 在 L4S 存储顺序中的位置为 B4/B3/B2，即 [3,2,1]；
# 其余 11 个通道为 B1、B5-B12 与坡度、高程。
L4S_RGB_CHANNELS = (3, 2, 1)
L4S_EXTRA_CHANNELS = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

STCA_IN_CHANNELS = len(L4S_EXTRA_CHANNELS)  # 11
STCA_EMBED_DIM = 96


class STCA(nn.Module):
    """额外通道 -> 逐 token、逐通道的乘性（或加性）调制。"""

    def __init__(self, in_chans=STCA_IN_CHANNELS, embed_dim=STCA_EMBED_DIM, fusion="mul"):
        super().__init__()
        if fusion not in {"mul", "add"}:
            raise ValueError(f"未知融合方式: {fusion}")
        self.fusion = fusion
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 均使用 PyTorch 默认初始化；只有末层显式置零。
        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=True,
        )
        self.ln = nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        self.zero = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        with torch.no_grad():
            self.zero.weight.zero_()
            self.zero.bias.zero_()

    def forward(self, x_extra, h0):
        """x_extra: (B,11,H,W) NCHW；h0: (B,H/p,W/p,96) NHWC。"""
        if x_extra.ndim != 4 or x_extra.shape[1] != self.in_chans:
            raise ValueError(
                f"额外通道输入应为 (B,{self.in_chans},H,W)，实际 {tuple(x_extra.shape)}"
            )
        e = F.avg_pool2d(x_extra, kernel_size=4, stride=4)
        if tuple(e.shape[-2:]) != tuple(h0.shape[1:3]):
            raise ValueError(
                f"池化网格 {tuple(e.shape[-2:])} 与 token 网格 {tuple(h0.shape[1:3])} 不一致"
            )
        u = self.conv1(e)
        u = self.dwconv(u)
        u = u.permute(0, 2, 3, 1)  # NCHW -> NHWC，与 h0 对齐
        u = F.gelu(self.ln(u))
        s = self.zero(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.fusion == "mul":
            return h0 * (1 + s)
        return h0 + s


class STCAPatchEmbed(nn.Module):
    """保留原 patch_embed 参数与 state_dict 键名，在输出后注入 STCA。

    只把 3 个物理 RGB 通道送入原 patch embedding；其余 11 个通道进入 STCA 支路。
    `proj`/`norm` 直接引用原模块的同一对象，因此参数身份和键名都不变，
    不复制任何已训练权重。
    """

    def __init__(
        self,
        patch_embed,
        adapter,
        rgb_channels=L4S_RGB_CHANNELS,
        extra_channels=L4S_EXTRA_CHANNELS,
    ):
        super().__init__()
        # 复制 timm PatchEmbed.forward 依赖的标量/元组属性
        self.img_size = patch_embed.img_size
        self.strict_img_size = patch_embed.strict_img_size
        self.dynamic_img_pad = patch_embed.dynamic_img_pad
        self.patch_size = patch_embed.patch_size
        self.flatten = patch_embed.flatten
        self.output_fmt = patch_embed.output_fmt
        # 共享同一批参数对象：state_dict 键名仍为 patch_embed.proj.* / patch_embed.norm.*
        self.proj = patch_embed.proj
        self.norm = patch_embed.norm
        self.adapter = adapter
        self.rgb_channels = tuple(rgb_channels)
        self.extra_channels = tuple(extra_channels)
        # 原模块的 forward 与原参数对象；不注册为子模块，避免 state_dict 键重复。
        self._rgb_forward = patch_embed.forward
        object.__setattr__(self, "_source_patch_embed", patch_embed)

    def forward(self, x):
        expect = len(self.rgb_channels) + len(self.extra_channels)
        if x.ndim != 4 or x.shape[1] != expect:
            raise ValueError(f"STCA 输入应为 (B,{expect},H,W)，实际 {tuple(x.shape)}")
        h0 = self._rgb_forward(x[:, list(self.rgb_channels)])
        return self.adapter(x[:, list(self.extra_channels)], h0)


def adapter_parameter_count(module):
    """适配器新增参数计数（只统计 STCA 支路）。"""
    return sum(p.numel() for p in module.parameters())

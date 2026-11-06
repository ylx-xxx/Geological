"""目标端输入放大版 SwinUPerNetL4S（新增文件，放入 train_l4s_qz/，不修改冻结代码）。

做法：数据、标签、评价全部保持原生 128x128；只在模型 forward 内部把输入双线性放大
`scale` 倍，Swin 在放大后的网格上运行。实际运算链（scale=2）：
    输入 128 -> 放大 256 -> Swin/UPerNet 融合网格 64 -> 父类 forward 插值到 256 -> 包装器插值回 128
state_dict 键名与 SwinUPerNetL4S 完全相同。

共享初始化（修订 A）：
    build_swin() 先按原 128 尺度构造参考模型（消耗随机数的方式与原流程逐位相同），
    再在保存/恢复 CPU 与 CUDA RNG 的上下文里构造 256 模型，
    最后把"同名且同形状"的未训练参数从参考复制到 256 模型。
    因此：解码器全部参数与原 128 模型逐位一致；训练随机起点（CPU/CUDA）与原流程一致；
    只有形状变化的两张相对位置偏置表（169x24 对 49x24）保持随机，随后由源权重迁移覆盖。
    不读任何已训练权重。
"""

import torch
import torch.nn.functional as F

from train_l4s_qz.model import SwinUPerNetL4S


class SwinUPerNetL4SScaled(SwinUPerNetL4S):
    def __init__(self, scale=2, base_size=128, **kwargs):
        if scale < 1:
            raise ValueError("scale must be >= 1")
        self.scale = int(scale)
        kwargs.pop("img_size", None)
        super().__init__(img_size=base_size * self.scale, **kwargs)

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.scale != 1:
            x = F.interpolate(
                x, scale_factor=self.scale, mode="bilinear", align_corners=False
            )
        y = super().forward(x)
        if self.scale != 1:
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return y


def build_swin(in_chans, base_size=128, scale=1):
    """训练端唯一构造入口。scale=1 与原 `SwinUPerNetL4S(in_chans, img_size)` 完全等价。"""
    reference = SwinUPerNetL4S(in_chans=in_chans, img_size=base_size)
    if scale == 1:
        reference.init_parity = None
        return reference

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model = SwinUPerNetL4SScaled(scale=scale, base_size=base_size, in_chans=in_chans)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    ref, own = reference.state_dict(), model.state_dict()
    if set(ref) != set(own):
        raise RuntimeError("scale 模型与参考模型的键集合不一致")
    resized = sorted(k for k in ref if ref[k].shape != own[k].shape)
    if not resized or any(not k.endswith("relative_position_bias_table") for k in resized):
        raise RuntimeError(f"形状变化的键不是预期的位置偏置表: {resized}")
    shared = {k: v for k, v in ref.items() if k not in resized}
    result = model.load_state_dict(shared, strict=False)
    if set(result.missing_keys) != set(resized) or result.unexpected_keys:
        raise RuntimeError("共享初始化复制的键不符合预期")
    model.init_parity = {
        "reference_built_at": base_size,
        "candidate_built_at": base_size * scale,
        "copied_keys": len(shared),
        "resized_keys": resized,
        "resized_elements_added": int(
            sum(own[k].numel() - ref[k].numel() for k in resized)
        ),
        "decoder_keys_all_copied": all(
            k in shared for k in ref if not k.startswith("encoder.")
        ),
    }
    del reference
    return model

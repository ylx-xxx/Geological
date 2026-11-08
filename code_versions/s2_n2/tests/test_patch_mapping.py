"""多通道初始化映射测试（I-03）。

覆盖三件事：
1. 历史 first3 路径数值不变（便于复现已登记结果）；
2. rgb 路径把源 R,G,B 核放到目标 [3,2,1]，其余通道按要求初始化；
3. 映射后的卷积在“非 RGB 通道为 0”的输入上，与 3 通道卷积输出一致。

运行：python -B -m unittest discover -s tests -p 'test_patch_mapping.py' -v
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_l4s_qz.model import (  # noqa: E402
    RGB_SOURCE_CHANNELS,
    RGB_TARGET_CHANNELS,
    adapt_patch_embed_weight,
)


def make_weights(src_ch=3, dst_ch=14, out_ch=4, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    old = torch.randn(out_ch, src_ch, k, k, generator=g)
    new = torch.randn(out_ch, dst_ch, k, k, generator=g)
    return old, new


class LegacyFirst3Test(unittest.TestCase):
    def test_legacy_positions_and_mean_extra(self):
        old, new = make_weights()
        out = adapt_patch_embed_weight(old, new, extra="mean", target_channels=None)
        # 前三个位置 = 源核原位复制（B1/B2/B3，即 I-03 的错配模式）
        self.assertTrue(torch.equal(out[:, :3], old[:, :3]))
        mean_w = old.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(out[:, 3:], mean_w.repeat(1, 11, 1, 1)))

    def test_target_not_mutated(self):
        old, new = make_weights()
        before = new.clone()
        adapt_patch_embed_weight(old, new, extra="zero", target_channels=None)
        self.assertTrue(torch.equal(new, before))


class RGBMappingTest(unittest.TestCase):
    def test_rgb_positions(self):
        old, new = make_weights()
        out = adapt_patch_embed_weight(
            old, new, extra="zero", target_channels=RGB_TARGET_CHANNELS
        )
        self.assertEqual(RGB_TARGET_CHANNELS, (3, 2, 1))
        for src, dst in zip(RGB_SOURCE_CHANNELS, RGB_TARGET_CHANNELS):
            self.assertTrue(torch.equal(out[:, dst], old[:, src]))

    def test_zero_extras(self):
        old, new = make_weights()
        out = adapt_patch_embed_weight(
            old, new, extra="zero", target_channels=RGB_TARGET_CHANNELS
        )
        extras = [i for i in range(14) if i not in RGB_TARGET_CHANNELS]
        self.assertTrue(torch.equal(out[:, extras], torch.zeros_like(out[:, extras])))

    def test_mean_extras(self):
        old, new = make_weights()
        out = adapt_patch_embed_weight(
            old, new, extra="mean", target_channels=RGB_TARGET_CHANNELS
        )
        extras = [i for i in range(14) if i not in RGB_TARGET_CHANNELS]
        mean_w = old.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(out[:, extras], mean_w.repeat(1, 11, 1, 1)))

    def test_random_extras_keep_target(self):
        old, new = make_weights()
        out = adapt_patch_embed_weight(
            old, new, extra="random", target_channels=RGB_TARGET_CHANNELS
        )
        extras = [i for i in range(14) if i not in RGB_TARGET_CHANNELS]
        self.assertTrue(torch.equal(out[:, extras], new[:, extras]))

    def test_invalid_mapping_rejected(self):
        old, new = make_weights()
        for bad in ((3, 2), (3, 3, 1), (3, 2, 14), (-1, 2, 1)):
            with self.assertRaises(ValueError):
                adapt_patch_embed_weight(
                    old, new, extra="zero", target_channels=bad
                )

    def test_unknown_extra_rejected(self):
        old, new = make_weights()
        with self.assertRaises(ValueError):
            adapt_patch_embed_weight(
                old, new, extra="garbage", target_channels=RGB_TARGET_CHANNELS
            )

    def test_shape_mismatch_returns_none(self):
        old, _ = make_weights(out_ch=4)
        _, new = make_weights(out_ch=5)
        self.assertIsNone(
            adapt_patch_embed_weight(old, new, target_channels=RGB_TARGET_CHANNELS)
        )

    def test_three_to_three_is_direct(self):
        old, new = make_weights(dst_ch=3)
        out = adapt_patch_embed_weight(
            old, new, extra="zero", target_channels=None
        )
        self.assertTrue(torch.equal(out, old))


class ConvEquivalenceTest(unittest.TestCase):
    """映射后的 14 通道卷积，在非 RGB 通道置零时与 3 通道卷积等价。"""

    def test_outputs_match(self):
        old, new = make_weights(out_ch=8, k=4, seed=1)
        mapped = adapt_patch_embed_weight(
            old, new, extra="zero", target_channels=RGB_TARGET_CHANNELS
        )
        conv3 = torch.nn.Conv2d(3, 8, 4, stride=4, bias=False)
        conv14 = torch.nn.Conv2d(14, 8, 4, stride=4, bias=False)
        with torch.no_grad():
            conv3.weight.copy_(old)
            conv14.weight.copy_(mapped)
        x3 = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))
        x14 = torch.zeros(2, 14, 32, 32)
        for src, dst in zip(RGB_SOURCE_CHANNELS, RGB_TARGET_CHANNELS):
            x14[:, dst] = x3[:, src]
        with torch.no_grad():
            y3 = conv3(x3)
            y14 = conv14(x14)
        # 仅存在浮点累加顺序差异
        self.assertLess(float((y3 - y14).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()

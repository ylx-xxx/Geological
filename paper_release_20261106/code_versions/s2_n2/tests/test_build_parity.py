"""模型构造一致性测试：同种子下，14 通道 rgb 映射模型与 3 通道模型共享参数相同。

运行：python -B -m unittest discover -s tests -p 'test_build_parity.py' -v
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproducibility.train import build_l4s_model  # noqa: E402
from train_l4s_qz.model import RGB_TARGET_CHANNELS  # noqa: E402

SEED = 2026


def patch_key(model):
    return next(
        k for k in model.state_dict() if k.endswith("patch_embed.proj.weight")
    )


class BuildParityTest(unittest.TestCase):
    def test_shared_params_identical(self):
        torch.manual_seed(SEED)
        m3 = build_l4s_model(3, "first3", img_size=128)
        torch.manual_seed(SEED)
        m14 = build_l4s_model(14, "rgb", img_size=128)
        key14 = patch_key(m14)
        a = m3.state_dict()
        b = m14.state_dict()
        self.assertEqual(set(a), set(b))
        diff = [k for k in a if k != key14 and not torch.equal(a[k], b[k])]
        self.assertEqual(diff, [])

    def test_patch_embed_shape_and_zeros(self):
        torch.manual_seed(SEED)
        m14 = build_l4s_model(14, "rgb", img_size=128)
        w = m14.state_dict()[patch_key(m14)]
        self.assertEqual(tuple(w.shape[:2]), (96, 14))
        extras = [i for i in range(14) if i not in RGB_TARGET_CHANNELS]
        self.assertTrue(torch.equal(w[:, extras], torch.zeros_like(w[:, extras])))

    def test_rgb_path_unchanged(self):
        # 3 通道输入走 rgb 映射时应与直接构造完全相同
        torch.manual_seed(SEED)
        a = build_l4s_model(3, "first3", img_size=128)
        torch.manual_seed(SEED)
        b = build_l4s_model(3, "rgb", img_size=128)
        for k in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[k], b.state_dict()[k]))


if __name__ == "__main__":
    unittest.main()

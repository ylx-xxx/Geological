import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, ignore_index=255, eps=1e-6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        valid = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid] = 0

        onehot = F.one_hot(target_safe, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()

        probs = probs * valid
        onehot = onehot * valid

        dims = (0, 2, 3)
        inter = torch.sum(probs * onehot, dims)
        union = torch.sum(probs + onehot, dims)

        dice = (2 * inter + self.eps) / (union + self.eps)
        loss_per_class = 1.0 - dice

        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            w = w / (w.mean() + self.eps)
            loss_per_class = loss_per_class * w

        return loss_per_class.mean()


class EmptyPatchBackgroundPenalty(nn.Module):
    """空标注 patch 的前景概率惩罚（本轮唯一新增训练项）。

    定义：对 batch 内每个“GT 无任何正类像素、但至少有一个有效背景像素”的 patch，
    取其背景像素上的前景概率 p1，按空标注 patch 的背景像素总数求平均。

        L_empty = (1 / N_empty_bg) * sum_{i in 空标注patch, y_i=0} p1_i

    细则（训练前写定，不得中途修改）：
    - 空标注只由 GT 定义：has_pos 为假且 has_bg 为真；不看模型预测。
    - ignore（255）像素既不进入分子也不进入分母；全 255 的 patch 不算空标注。
    - batch 内没有空标注像素时该项恒为 0（divide-by-clamp，不是 NaN）。
    - 归一化按空标注 patch 内的背景像素数，不按 batch 总数、也不按 patch 数。
    - 不使用原类别权重；该惩罚是对空标注区域的额外约束，不是对全部背景重新加权。
    """

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)[:, 1]
        valid = target != self.ignore_index
        has_pos = ((target == 1) & valid).flatten(1).any(dim=1)
        has_bg = ((target == 0) & valid).flatten(1).any(dim=1)
        empty = (~has_pos) & has_bg
        mask = (target == 0) & empty[:, None, None]
        # 用 clamp 而不是 .item()，避免每步同步；无空标注时分子为 0，结果恰为 0。
        denom = mask.sum().clamp(min=1)
        return (probs * mask).sum() / denom


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes=2,
        ignore_index=255,
        ce_weight=1.0,
        dice_weight=1.0,
        class_weights=None,
        label_smoothing=0.02,
        empty_penalty_weight=0.0,
    ):
        super().__init__()

        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_weights_tensor = None

        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=class_weights_tensor,
            label_smoothing=label_smoothing,
        )

        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        # 默认 0.0 时完全跳过新项，保证与历史损失逐位一致。
        self.empty_penalty_weight = float(empty_penalty_weight)
        self.empty_penalty = (
            EmptyPatchBackgroundPenalty(ignore_index=ignore_index)
            if self.empty_penalty_weight > 0
            else None
        )

    def forward(self, logits, target):
        loss = self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
        if self.empty_penalty is not None:
            loss = loss + self.empty_penalty_weight * self.empty_penalty(logits, target)
        return loss

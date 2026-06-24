import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        # クラス重み (num_classes,)。None なら均等。
        self.register_buffer("weight", weight)

    def forward(self, logits, soft_targets):

        log_probs = F.log_softmax(logits, dim=1)

        if self.weight is not None:
            # クラスごとに重み付け（ターゲット分布の各クラスのlossをスケール）
            log_probs = log_probs * self.weight

        loss = -(soft_targets * log_probs).sum(dim=1)

        return loss.mean()
    
def build_soft_target(answers, num_classes, ignore_index=None):
    """
    answers:
        (batch_size, 10)

    return:
        (batch_size, num_classes)
    """

    batch_size = answers.shape[0]

    target = torch.zeros(
        batch_size,
        num_classes,
        device=answers.device,
    )

    for i in range(batch_size):
        valid_count = 0

        for ans in answers[i]:
            ans_idx = int(ans)

            if ignore_index is not None and ans_idx == ignore_index:
                continue

            target[i, ans_idx] += 1
            valid_count += 1

        # 全て ignore 対象なら元の分布でフォールバック
        if valid_count == 0:
            for ans in answers[i]:
                target[i, int(ans)] += 1
            valid_count = answers.shape[1]

        target[i] /= valid_count

    return target


class VQABCELoss(nn.Module):
    """VQA 標準の BCE 損失（BUTD/LXMERT 系）。

    各クラスを独立に sigmoid し、soft score（min(投票数/3,1)）との binary CE。
    分布(softmax)ではなく「各回答が正解か」を独立に当てる定式化で、
    VQA の部分点採点に直結するため honest が伸びやすい。
    クラス方向に和、バッチ方向に平均（勾配スケールを CE と同程度に保つ）。
    weight があればクラスごとに loss をスケール（CLASS_WEIGHTS 反映）。
    """

    def __init__(self, weight=None):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, logits, soft_scores):
        bce = F.binary_cross_entropy_with_logits(
            logits, soft_scores, reduction="none")  # (B, C)
        if self.weight is not None:
            bce = bce * self.weight
        return bce.sum(dim=1).mean()


def build_vqa_score_target(answers, num_classes):
    """VQA soft score ターゲット (B, num_classes)。各クラス = min(投票数/3, 1)。
    soft target（合計1の分布）とは違い、合計1にしない（独立な多ラベル）。"""
    from collections import Counter
    B = answers.shape[0]
    target = torch.zeros(B, num_classes, device=answers.device)
    for i in range(B):
        for idx, cnt in Counter(int(a) for a in answers[i]).items():
            target[i, idx] = min(cnt / 3.0, 1.0)
    return target
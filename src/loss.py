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
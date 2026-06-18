import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):

    def forward(self, logits, soft_targets):

        log_probs = F.log_softmax(logits, dim=1)

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
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


def _combined_cols(answers_i, ans_ocr_pos_i, n_answer):
    """1サンプルの10回答を結合出力空間の列番号に写す。
    OCRコピー位置(pos>=0)なら n_answer+pos、そうでなければ語彙idx。"""
    cols = []
    for ans, pos in zip(answers_i, ans_ocr_pos_i):
        p = int(pos)
        cols.append(n_answer + p if p >= 0 else int(ans))
    return cols


def build_combined_soft_target(answers, ans_ocr_pos, n_answer, num_ocr,
                               ignore_index=None):
    """OCRコピー込みの soft target (B, n_answer+num_ocr)。
    各回答が「語彙外かつOCR一致」なら OCR列、それ以外は語彙列に質量を置く。"""
    B = answers.shape[0]
    target = torch.zeros(B, n_answer + num_ocr, device=answers.device)
    for i in range(B):
        cols = _combined_cols(answers[i], ans_ocr_pos[i], n_answer)
        valid = [c for c in cols
                 if not (ignore_index is not None and c == ignore_index)]
        if not valid:  # 全て ignore ならフォールバック
            valid = cols
        for c in valid:
            target[i, c] += 1
        target[i] /= len(valid)
    return target


def build_combined_hard_target(answers, ans_ocr_pos, n_answer):
    """OCRコピー込みの hard target (B,)。結合列番号の最頻値。"""
    from collections import Counter
    B = answers.shape[0]
    out = torch.zeros(B, dtype=torch.long, device=answers.device)
    for i in range(B):
        cols = _combined_cols(answers[i], ans_ocr_pos[i], n_answer)
        out[i] = Counter(cols).most_common(1)[0][0]
    return out
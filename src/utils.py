import random
import time

import numpy as np
import torch

from torchvision import transforms

from configs.baseline import (
    IMAGE_SIZE, NORMALIZE, NORM_MEAN, NORM_STD, AUGMENT, IMAGE_BACKBONE,
)

# CLIP の前処理は専用の正規化統計（ImageNet とは別）。CLIP使用時はこちらを使う。
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_transform(train=False, strong=False):
    """画像前処理を返す。

    - train=False（既定）: Resize→ToTensor(→Normalize)。inference/analyze 用。
      推論はランダム性を入れたくないので必ずこれ。
    - train=True かつ AUGMENT=True: 全データ共通の「軽い aug」を付与。
      色を答える質問が多いので hue/saturation は揺らさない（色回答の保護）。
    - train=True かつ strong=True: 少数クラス向けの「強い aug」を上乗せ
      （ブラー・遠近変形）。VizWiz のブレ/構図崩れにも効く。

    NORMALIZE で Normalize の有無を切替。学習時と推論時で Resize/Normalize
    が共通なので分布ズレは起きない（変わるのは aug の有無のみ）。
    """
    if train and AUGMENT:
        # 軽い aug（全データ共通）
        ops = [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # hue/sat はゼロ
        ]
        if strong:
            # 少数クラス向けの追加 aug（より強い変形で多様性を稼ぐ）
            ops += [
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5)], p=0.4
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            ]
    else:
        ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    ops.append(transforms.ToTensor())
    if NORMALIZE:
        # CLIP backbone のときは CLIP 専用統計、それ以外は config の統計。
        if IMAGE_BACKBONE.startswith("clip"):
            mean, std = CLIP_MEAN, CLIP_STD
        else:
            mean, std = NORM_MEAN, NORM_STD
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def question_type(q):
    """正規化済み質問文 q を粗い意味タイプに分類する。
    質問タイプ別の unanswerable 補正（推論）や診断の集計に使う。"""
    w = q.split(" ") if q else []
    head = w[0] if w else ""
    head2 = " ".join(w[:2])
    if "color" in q or "colour" in q:
        return "color"
    if head2 in ("how many", "how much") or "how many" in q:
        return "count"
    if "brand" in q:
        return "brand"
    if head in ("is", "are", "was", "were", "do", "does", "did",
                "can", "could", "will", "would", "should", "has", "have"):
        return "yes/no"
    if head == "what":
        return "what(other)"
    if head in ("where", "who", "when", "why", "which", "how"):
        return head
    return "other"


def apply_unanswerable_bias(logits, qtexts, unanswerable_idx,
                            by_qtype, default_bias=0.0):
    """質問タイプ別に unanswerable の logit を調整する（推論時のみ。学習は不変）。

    logits[i, unanswerable_idx] += bias を行う。
      bias > 0: その質問タイプで unanswerable を出しやすく（answerable率が低い count 等）
      bias < 0: 出しにくく（answerable率が高い color 等）
    bias は by_qtype[type] を引き、無ければ default_bias。

    Parameters
    ----------
    logits : torch.Tensor  (N, C)  ※ in-place で書き換える
    qtexts : list[str]     process_text 済みの質問文（len==N）
    unanswerable_idx : int | None  None なら何もしない
    by_qtype : dict        {質問タイプ: bias}
    default_bias : float   未指定タイプに使う bias
    """
    if unanswerable_idx is None:
        return logits
    for i, q in enumerate(qtexts):
        b = by_qtype.get(question_type(q), default_bias)
        if b:
            logits[i, unanswerable_idx] += b
    return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
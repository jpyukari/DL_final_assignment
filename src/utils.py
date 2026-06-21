import random
import time

import numpy as np
import torch

from torchvision import transforms

from configs.baseline import (
    IMAGE_SIZE, NORMALIZE, NORM_MEAN, NORM_STD, AUGMENT,
)


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
        ops.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))
    return transforms.Compose(ops)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import random
import time

import numpy as np
import torch

from torchvision import transforms

from configs.baseline import IMAGE_SIZE, NORMALIZE, NORM_MEAN, NORM_STD


def build_transform():
    """train / inference / analyze 共通の画像前処理を返す。

    NORMALIZE フラグで Normalize の有無を切り替える。3 箇所で同じ
    前処理を使うことで、学習時と推論時の分布ズレを防ぐ。
    """
    ops = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
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
"""
train 画像から正規化用の mean/std を実測する。

スクラッチ学習なので ImageNet 統計よりデータセット実測値の方が望ましい。
出力された値を configs/baseline.py の NORM_MEAN / NORM_STD に貼る。

実行:
  python -m src.compute_stats
  python -m src.compute_stats --max-images 5000   # 一部だけで概算
"""
import argparse

import torch
from torchvision import transforms

from configs.baseline import IMAGE_SIZE
from src.dataset import VQADataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", default="./data/train_split.json")
    parser.add_argument("--image-dir", default="./data/train")
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="先頭 N 枚だけで概算（未指定なら全件）",
    )
    args = parser.parse_args()

    # Normalize なしの素の前処理（ToTensor で 0..1）
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = VQADataset(
        df_path=args.df,
        image_dir=args.image_dir,
        transform=transform,
        answer=False,
    )

    n = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))

    # チャンネルごとに sum / sum(x^2) を貯めて mean/std を出す
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    pixel_count = 0

    for i in range(n):
        img = dataset[i]["image"]              # (3, H, W), 0..1
        channel_sum += img.sum(dim=[1, 2])
        channel_sq_sum += (img ** 2).sum(dim=[1, 2])
        pixel_count += img.shape[1] * img.shape[2]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")

    mean = channel_sum / pixel_count
    std = (channel_sq_sum / pixel_count - mean ** 2).clamp_min(0).sqrt()

    mean = [round(v, 4) for v in mean.tolist()]
    std = [round(v, 4) for v in std.tolist()]

    print(f"\n{n} 枚で計算:")
    print(f"NORM_MEAN = {mean}")
    print(f"NORM_STD = {std}")
    print("\n^ この2行を configs/baseline.py に貼り付けてください。")


if __name__ == "__main__":
    main()

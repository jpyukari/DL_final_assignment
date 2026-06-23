"""
自己教師あり事前学習（SimSiam）でバックボーン ResNet の普遍的特徴を獲得する。

ルール: タスク（VQA）を解くための学習は不可。ラベルを一切使わない
自己教師あり学習のみ。本スクリプトは画像のみ（回答ラベル不使用）で学習する。

データ: 既定で VizWiz の train+test 画像（ラベル無し画像として使用）。
ドメイン一致が最重要なので自前画像が最適。規模が欲しければ --extra-dirs で
OpenImages 等の外部画像フォルダを足せる（SSL=汎用特徴獲得なので許可範囲）。

拡張の方針: 色相/彩度は揺らさない・グレースケール化しない。SSL は拡張不変性を
学ぶため、強い色拡張は「色不変」特徴を生み、色を答える質問を悪化させる。
→ 幾何変形 + ブラー（VizWiz のブレに対応）+ 明度/コントラストのみ。

生成物: ./outputs/checkpoints/ssl_backbone.pt（VQAModel.resnet にロードできる重み）
使い方:
  python -m src.ssl_pretrain                       # train+valid 画像で SSL
  python -m src.ssl_pretrain --epochs 100 --batch-size 256
  python -m src.ssl_pretrain --extra-dirs ./data/openimages
そのあと configs/baseline.py の PRETRAINED_BACKBONE にパスを設定して train.py。
"""
import os
import glob
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from configs.baseline import RESNET, NORMALIZE, NORM_MEAN, NORM_STD
from src.models.resnet import ResNet18, ResNet34, ResNet50

RESNET_FACTORY = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
}

IMAGE_EXT = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def build_ssl_transform(img_size):
    """色保持の SSL 拡張（hue/saturation は触らない・grayscale 無し）。"""
    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4)],  # hue/sat=0
            p=0.7,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.5,
        ),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]
    if NORMALIZE:
        ops.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))
    return transforms.Compose(ops)


class TwoViewDataset(torch.utils.data.Dataset):
    """画像1枚から拡張済み2ビューを返す（ラベル無し）。"""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)


def gather_image_paths(dirs):
    paths = []
    for d in dirs:
        for ext in IMAGE_EXT:
            paths.extend(glob.glob(os.path.join(d, ext)))
    return sorted(set(paths))


class SimSiam(nn.Module):
    """SimSiam: backbone(512次元) + projector + predictor、stop-grad で崩壊回避。"""

    def __init__(self, backbone, dim=512, pred_dim=128):
        super().__init__()
        self.backbone = backbone  # ResNet（最終 fc が 512 次元を出力）
        self.projector = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True), nn.Linear(pred_dim, dim),
        )

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def neg_cosine(p, z):
    return -F.cosine_similarity(p, z, dim=1).mean()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("GPU (CUDA/MPS) が利用できません。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+",
                        default=["./data/train", "./data/valid"],
                        help="SSL に使う画像フォルダ（ラベル不使用）")
    parser.add_argument("--extra-dirs", nargs="+", default=[],
                        help="外部画像フォルダを追加（OpenImages 等）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=None,
                        help="未指定なら 0.05*batch/256")
    parser.add_argument("--out", default="./outputs/checkpoints/ssl_backbone.pt")
    args = parser.parse_args()

    device = get_device()
    print("device =", device)

    paths = gather_image_paths(args.dirs + args.extra_dirs)
    if not paths:
        raise RuntimeError(f"画像が見つかりません: {args.dirs + args.extra_dirs}")
    print(f"SSL 画像枚数: {len(paths)}  (dirs={args.dirs + args.extra_dirs})")

    dataset = TwoViewDataset(paths, build_ssl_transform(args.img_size))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    backbone = RESNET_FACTORY[RESNET]()
    model = SimSiam(backbone).to(device)

    base_lr = args.lr if args.lr is not None else 0.05 * args.batch_size / 256
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        # cosine 学習率スケジュール
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"ssl [{epoch+1}/{args.epochs}] lr={lr:.4f}")
        for x1, x2 in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            p1, p2, z1, z2 = model(x1, x2)
            loss = 0.5 * neg_cosine(p1, z2) + 0.5 * neg_cosine(p2, z1)

            if not torch.isfinite(loss):
                raise RuntimeError(f"loss が NaN/Inf（epoch {epoch+1}）。lr を下げてください。")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total / len(loader)
        # loss は -1 に近いほど良い（cosine 類似度が高い）
        print(f"Epoch [{epoch+1}/{args.epochs}] SSL loss={avg:.4f}")

        # 毎エポック backbone（resnet 部分のみ）を保存。VQAModel.resnet にロード可能。
        torch.save(model.backbone.state_dict(), args.out)

    print(f"saved backbone: {args.out}")
    print("→ configs/baseline.py の PRETRAINED_BACKBONE に設定して train.py")


if __name__ == "__main__":
    main()

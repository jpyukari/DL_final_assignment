"""
診断スクリプト(B): 「画像だけ」で特定タイプ（既定: color）の質問に答えられるかを測る。

背景: 質問だけモデル(src.question_only)で color は honest acc=0.26 と低かったが、
無学習の「平均色」ヒューリスティックは 0.11〜0.15 で多数派(unanswerable=0.24)に負けた。
平均色は画像全体を濁色に潰すので粗すぎる＝「CNN が色を取り出せるか」は別途学習が要る。
本スクリプトは質問を一切使わず画像→回答だけを学習して、それを測る。

読み方:
  - 画像だけ acc >> 多数派ベースライン
        → 画像は色情報を持つ。問題は「融合（画像を答えに効かせる）」。
          → cross_attention(FUSION) に投資する価値が高い。
  - 画像だけ acc ≈ 多数派ベースライン
        → 画像から色が取り出せていない。backbone/前処理/正規化が疑わしい。
          → 融合を変えても無駄。特徴抽出側を直す。
  併せて「answerable率（多数派が unanswerable でない割合）」も出し、
  そもそもの伸びしろ（実際に色が答えになっている割合）を把握する。

対象タイプは config の PROBE_QTYPE で切替（color / count / ...）。

実行:
  python -m src.color_probe                         # PROBE_QTYPE を probe
  python -m src.color_probe --qtype count --epochs 10 --img-size 160
"""
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from configs.baseline import (
    RESNET, PROBE_QTYPE, PRETRAINED_BACKBONE, SEED, BATCH_SIZE,
    NORMALIZE, NORM_MEAN, NORM_STD,
)
from src.dataset import VQADataset, process_text, UNK_ANSWER
from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.metrics import vqa_acc_string
from src.question_only import question_type
from src.utils import set_seed

from torchvision import transforms

RESNET_FACTORY = {"resnet18": ResNet18, "resnet34": ResNet34, "resnet50": ResNet50}


def probe_transform(img_size):
    """probe 用の軽い前処理（aug 無し・config と同じ正規化）。"""
    ops = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if NORMALIZE:
        ops.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))
    return transforms.Compose(ops)


class ImageOnlyModel(nn.Module):
    """画像だけ → 回答。質問を使わない（画像の情報量を測るため）。"""

    def __init__(self, backbone, n_answer):
        super().__init__()
        self.resnet = RESNET_FACTORY[backbone]()  # 512 次元出力
        self.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, n_answer),
        )

    def forward(self, image):
        return self.fc(self.resnet(image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtype", default=PROBE_QTYPE)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=160,
                        help="probe 用。速度優先で既定は小さめ")
    args = parser.parse_args()

    set_seed(SEED)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device={device}  qtype={args.qtype}  backbone={RESNET}  img={args.img_size}")

    tf = probe_transform(args.img_size)
    train_ds = VQADataset(df_path="./data/train_split.json",
                          image_dir="./data/train", transform=tf)
    valid_ds = VQADataset(df_path="./data/valid_split.json",
                          image_dir="./data/train", transform=tf)
    valid_ds.update_dict(train_ds)
    a2i, idx2answer = train_ds.answer2idx, train_ds.idx2answer
    n_answer = len(a2i)

    # 対象タイプの index を抽出
    def idxs_of(ds):
        return [i for i in range(len(ds.df))
                if question_type(process_text(ds.df["question"][i])) == args.qtype]
    tr_idx, va_idx = idxs_of(train_ds), idxs_of(valid_ds)
    print(f"対象サンプル: train={len(tr_idx)}  valid={len(va_idx)}")
    if not tr_idx or not va_idx:
        raise SystemExit(f"qtype='{args.qtype}' のサンプルが足りません。")

    # 伸びしろ把握: valid 対象のうち多数派回答が unanswerable でない割合
    def majority_str(i, ds):
        c = Counter(process_text(a["answer"]) for a in ds.df["answers"][i])
        return c.most_common(1)[0][0]
    answerable = sum(majority_str(i, valid_ds) != "unanswerable" for i in va_idx)
    print(f"answerable率(多数派≠unanswerable): {answerable}/{len(va_idx)} "
          f"= {answerable/len(va_idx):.2%}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_ds, tr_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(valid_ds, va_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 多数派ベースライン: train 対象サンプルの最頻 mode_answer を valid 全部に出す
    tr_modes = Counter(int(train_ds[i]["mode_answer"]) for i in tr_idx[:2000])
    majority_idx = tr_modes.most_common(1)[0][0]
    maj_str = idx2answer[majority_idx]
    maj_str = "unanswerable" if maj_str == UNK_ANSWER else maj_str
    maj_acc = float(np.mean([
        vqa_acc_string(maj_str, [process_text(a["answer"])
                                 for a in valid_ds.df["answers"][i]])
        for i in va_idx
    ]))
    print(f"多数派ベースライン('{maj_str}'を全部): honest acc={maj_acc:.4f}")

    model = ImageOnlyModel(RESNET, n_answer).to(device)
    if PRETRAINED_BACKBONE:
        try:
            sd = torch.load(PRETRAINED_BACKBONE, map_location="cpu")
            miss, unexp = model.resnet.load_state_dict(sd, strict=False)
            print(f"[SSL] backbone ロード: {PRETRAINED_BACKBONE} "
                  f"(missing={len(miss)}, unexpected={len(unexp)})")
        except FileNotFoundError:
            print(f"[SSL] {PRETRAINED_BACKBONE} が無いのでスクラッチ")

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            y = batch["mode_answer"].to(device)
            logits = model(img)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(y)

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in valid_loader:
                preds.extend(model(batch["image"].to(device)).argmax(1).cpu().tolist())
        honest = []
        for i, p in zip(va_idx, preds):
            ps = idx2answer[p]
            ps = "unanswerable" if ps == UNK_ANSWER else ps
            gts = [process_text(a["answer"]) for a in valid_ds.df["answers"][i]]
            honest.append(vqa_acc_string(ps, gts))
        acc = float(np.mean(honest))
        best = max(best, acc)
        print(f"Epoch [{ep+1}/{args.epochs}] train_loss={total/len(tr_idx):.4f} "
              f"画像だけ honest acc={acc:.4f} distinct_preds={len(set(preds))}")

    print(f"\n=== 結論 (qtype={args.qtype}) ===")
    print(f"  answerable率       : {answerable/len(va_idx):.2%} (実際に答えが付く割合)")
    print(f"  多数派ベースライン  : {maj_acc:.4f}")
    print(f"  質問だけ(参考)      : 0.26  ※ src.question_only の color")
    print(f"  画像だけ(best)      : {best:.4f}")
    if best > maj_acc + 0.03:
        print("  → 画像 > ベースライン: 画像に色情報あり。融合(cross_attention)に投資価値。")
    else:
        print("  → 画像 ≈ ベースライン: 画像から色を取り出せていない。特徴抽出側を疑う。")


if __name__ == "__main__":
    main()

"""
推論時の unanswerable logit 補正を「質問タイプ別」に valid_split で自動チューニング。

背景: answerable率はタイプで両極端（count≈11% / color≈85%）。よって最適な
unanswerable バイアスもタイプごとに違う。各タイプ独立に honest acc（LB相当）を
最大化する bias を探索し、configs に貼れる UNANSWERABLE_BIAS_BY_QTYPE を出力する。

評価は honest acc（<unk>→unanswerable 変換＋元文字列照合）。index 照合だと
<unk> 同士一致で過大評価され public とズレるため。

実行:
  python -m src.sweep_unanswerable
  python -m src.sweep_unanswerable --model ./outputs/checkpoints/model.pt --resnet resnet34
  python -m src.sweep_unanswerable --biases -4 -3 -2 -1 0 1 2 3 4
"""
import argparse
from collections import defaultdict

import numpy as np
import torch

from configs.baseline import *

from src.dataset import VQADataset, process_text, UNK_ANSWER
from src.metrics import vqa_acc_string
from src.models.baseline import VQAModel
from src.utils import build_transform, question_type


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("GPU (CUDA/MPS) が利用できません。")


@torch.no_grad()
def collect_logits(model, dataset, device):
    """valid_split 全件の logits を集める（順序は df と一致, shuffle=False）。"""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    logits_all = []
    for batch in loader:
        image = batch["image"].to(device)
        question = batch["question"].to(device)
        logits_all.append(model(image, question).cpu())
    return torch.cat(logits_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--resnet", default=None,
                        help="config の RESNET を上書き（古い checkpoint 用）")
    parser.add_argument("--biases", type=float, nargs="+",
                        default=[-4, -3, -2, -1.5, -1, -0.5, 0,
                                 0.5, 1, 1.5, 2, 3, 4])
    parser.add_argument("--min-support", type=int, default=5,
                        help="これ未満のタイプは bias を振らず default(0) 据え置き")
    args = parser.parse_args()

    device = get_device()
    print("device =", device)
    resnet = args.resnet or RESNET

    transform = build_transform()
    train_dataset = VQADataset(df_path="./data/train_split.json",
                               image_dir="./data/train", transform=transform)
    valid_dataset = VQADataset(df_path="./data/valid_split.json",
                               image_dir="./data/train", transform=transform)
    valid_dataset.update_dict(train_dataset)

    idx2answer = train_dataset.idx2answer
    unans = train_dataset.answer2idx.get("unanswerable")
    if unans is None:
        raise RuntimeError("answer2idx に 'unanswerable' がありません。")

    model = VQAModel(
        vocab_size=len(train_dataset.question2idx) + 1,
        n_answer=len(train_dataset.answer2idx),
        backbone=resnet,
    ).to(device)
    state = torch.load(args.model, map_location=device)
    bad = [k for k, v in state.items()
           if torch.is_tensor(v) and not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(
            f"checkpoint に NaN/Inf があります（{len(bad)} tensors）: {args.model}")
    model.load_state_dict(state)
    model.eval()

    print("collecting logits on valid_split...")
    logits = collect_logits(model, valid_dataset, device)  # (N, C)
    N = logits.shape[0]

    # 文字列化（<unk>→unanswerable）と GT・タイプを前計算
    pred_str_of = [
        "unanswerable" if idx2answer[i] == UNK_ANSWER else idx2answer[i]
        for i in range(len(idx2answer))
    ]
    gt_strings = [
        [process_text(a["answer"]) for a in valid_dataset.df["answers"][i]]
        for i in range(N)
    ]
    qtypes = [question_type(process_text(valid_dataset.df["question"][i]))
              for i in range(N)]

    # タイプ別にサンプル index をまとめる
    by_type = defaultdict(list)
    for i, t in enumerate(qtypes):
        by_type[t].append(i)

    def honest_for(indices, bias):
        """indices の honest acc を、unanswerable に bias を足して評価。"""
        if not indices:
            return 0.0, 0
        sub = logits[indices].clone()
        sub[:, unans] += bias
        preds = sub.argmax(1).tolist()
        accs = [vqa_acc_string(pred_str_of[p], gt_strings[idx])
                for p, idx in zip(preds, indices)]
        return float(np.mean(accs)), len(indices)

    # 全体ベースライン（bias 0）
    base_acc, _ = honest_for(list(range(N)), 0.0)

    print(f"\n{'qtype':>12} | {'n':>4} | {'acc@0':>6} | {'best':>5} | "
          f"{'acc@best':>8} | gain")
    print("-" * 58)

    best_bias = {}
    chosen_total = 0.0  # support 重み付き honest（best 適用後）
    for t in sorted(by_type, key=lambda k: -len(by_type[k])):
        idxs = by_type[t]
        acc0, n = honest_for(idxs, 0.0)
        if n < args.min_support:
            # 少数タイプは default(0) 据え置き（過学習防止）
            chosen_total += acc0 * n
            print(f"{t:>12} | {n:>4} | {acc0:>6.3f} | {'--':>5} | "
                  f"{acc0:>8.3f} | (support<{args.min_support})")
            continue
        # このタイプだけ bias を振って best を選ぶ
        cand = [(b, honest_for(idxs, b)[0]) for b in args.biases]
        bb, ba = max(cand, key=lambda x: x[1])
        if bb != 0.0 and ba > acc0:
            best_bias[t] = bb
        chosen_total += ba * n
        flag = "" if bb == 0 else f"  +{ba-acc0:.3f}"
        print(f"{t:>12} | {n:>4} | {acc0:>6.3f} | {bb:>5.1f} | "
              f"{ba:>8.3f} |{flag}")

    tuned_acc = chosen_total / N

    print(f"\n全体 honest acc: {base_acc:.4f} (bias無) → {tuned_acc:.4f} "
          f"(タイプ別best)   gain=+{tuned_acc-base_acc:.4f}")
    print("\n--- configs/baseline.py に貼る ---")
    print("UNANSWERABLE_BIAS_BY_QTYPE = {")
    for t, b in best_bias.items():
        print(f"    {t!r}: {b},")
    print("}")
    print("UNANSWERABLE_BIAS_DEFAULT = 0.0")


if __name__ == "__main__":
    main()

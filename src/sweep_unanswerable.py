"""
推論時の unanswerable logit 補正量を valid_split で選ぶスクリプト。

学習はそのままに、推論時 logit_unanswerable -= delta して
VQA スコアがどう変わるかを delta ごとに比較する（GT があるので
リーダーボードに投げずに最適 delta を選べる）。

実行:
  python -m src.sweep_unanswerable
  python -m src.sweep_unanswerable --deltas 0 0.2 0.5 1.0 1.5 2.0
  python -m src.sweep_unanswerable --model ./outputs/checkpoints/model.pt --resnet resnet34
"""
import argparse

import torch

from configs.baseline import *

from src.dataset import VQADataset
from src.metrics import VQA_criterion
from src.models.baseline import VQAModel
from src.utils import build_transform


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("GPU (CUDA/MPS) が利用できません。")


@torch.no_grad()
def collect_logits(model, dataset, device):
    """valid_split 全件の logits と 10 人分の回答を集める。"""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    logits_all, answers_all = [], []
    for batch in loader:
        image = batch["image"].to(device)
        question = batch["question"].to(device)
        logits = model(image, question)
        logits_all.append(logits.cpu())
        answers_all.append(batch["answers"])
    return torch.cat(logits_all), torch.cat(answers_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--resnet", default=None,
                        help="config の RESNET を上書き（古い checkpoint 用）")
    parser.add_argument("--deltas", type=float, nargs="+",
                        default=[0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
    args = parser.parse_args()

    device = get_device()
    print("device =", device)
    resnet = args.resnet or RESNET

    transform = build_transform()

    train_dataset = VQADataset(
        df_path="./data/train_split.json",
        image_dir="./data/train",
        transform=transform,
    )
    valid_dataset = VQADataset(
        df_path="./data/valid_split.json",
        image_dir="./data/train",
        transform=transform,
    )
    valid_dataset.update_dict(train_dataset)

    unanswerable_idx = train_dataset.answer2idx.get("unanswerable")
    if unanswerable_idx is None:
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
            f"checkpoint に NaN/Inf があります（{len(bad)} tensors）: {args.model}\n"
            "発散したモデルです。再学習してください。"
        )
    model.load_state_dict(state)
    model.eval()

    print("collecting logits on valid_split...")
    logits, answers = collect_logits(model, valid_dataset, device)
    n = logits.shape[0]

    print(f"\n{'delta':>6} | {'VQA acc':>8} | {'unans予測':>9} | 比率")
    print("-" * 40)
    best = None
    for delta in args.deltas:
        adj = logits.clone()
        adj[:, unanswerable_idx] -= delta
        preds = adj.argmax(1)
        acc = VQA_criterion(preds, answers)
        n_unans = int((preds == unanswerable_idx).sum())
        ratio = n_unans / n
        mark = ""
        if best is None or acc > best[1]:
            best = (delta, acc)
            mark = "  <-- best"
        print(f"{delta:>6.2f} | {acc:>8.4f} | {n_unans:>9d} | {ratio:>5.1%}{mark}")

    print(f"\nbest delta = {best[0]} (valid VQA acc = {best[1]:.4f})")
    print("→ configs/baseline.py の UNANSWERABLE_LOGIT_BIAS に設定して "
          "python inference.py")


if __name__ == "__main__":
    main()

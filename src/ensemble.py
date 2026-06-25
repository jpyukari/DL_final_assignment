"""
複数チェックポイントの「確率平均」アンサンブル → タイプ別 unanswerable 自動補正
→ 提出zip + レポート。

異なる backbone（CLIP-B / ViT-L）を混ぜると間違え方が多様になり効きやすい。
valid に依存しない本物の汎化改善なので、public が止まった局面の堅い一手。
※ CLIP系backbone専用（224px・CLIP正規化が共通のため）。

各 run の best_model.pt を名前付きで退避してから使う:
  cp outputs/checkpoints/best_model.pt outputs/checkpoints/clipb.pt   # CLIP-B 学習後
  cp outputs/checkpoints/best_model.pt outputs/checkpoints/vitl.pt    # ViT-L 学習後

実行:
  python -m src.ensemble \
    --models outputs/checkpoints/clipb.pt:clip_vit_b_16 \
             outputs/checkpoints/vitl.pt:clip_vit_l_14
"""
import os
import argparse
from collections import defaultdict
from zipfile import ZipFile
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from configs.baseline import *  # noqa
import src.models.baseline as M
from src.models.baseline import VQAModel
from src.dataset import VQADataset, process_text, UNK_ANSWER
from src.utils import build_transform, question_type
from src.metrics import vqa_acc_string


def decode(p, idx2answer):
    """予測idx→回答文字列。<unk> は提出と同じく unanswerable に変換。"""
    s = idx2answer[p]
    return "unanswerable" if s == UNK_ANSWER else s


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("GPU (CUDA/MPS) が利用できません。")


@torch.no_grad()
def model_probs(ckpt, backbone, train_ds, dsets, device):
    """1モデルを構築・ロードし、各データセットの softmax 確率を返す。
    backbone ごとに IMAGE_BACKBONE を差し替えて構築（CLIP系のみ）。"""
    M.IMAGE_BACKBONE = backbone  # VQAModel.__init__ が参照する種別を差し替え
    model = VQAModel(
        vocab_size=len(train_ds.question2idx) + 1,
        n_answer=len(train_ds.answer2idx),
        backbone=RESNET,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  loaded {ckpt} as {backbone}")

    out = {}
    for name, ds in dsets.items():
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        probs = []
        for batch in loader:
            logits = model(batch["image"].to(device), batch["question"].to(device))
            probs.append(F.softmax(logits, dim=1).cpu())
        out[name] = torch.cat(probs)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


DEFAULT_BIASES = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]


def sweep_biases_on_logits(logits, valid_ds, idx2answer, unans, biases=None):
    """平均確率(の log)を logits として、タイプ別 unanswerable bias を探索。"""
    biases = biases or DEFAULT_BIASES
    N = logits.shape[0]
    gt = [[process_text(a["answer"]) for a in valid_ds.df["answers"][i]]
          for i in range(N)]
    qtypes = [question_type(process_text(valid_ds.df["question"][i]))
              for i in range(N)]
    by_type = defaultdict(list)
    for i, t in enumerate(qtypes):
        by_type[t].append(i)

    def honest_for(idxs, bias):
        if not idxs:
            return 0.0
        sub = logits[idxs].clone()
        sub[:, unans] += bias
        preds = sub.argmax(1).tolist()
        return float(np.mean([
            vqa_acc_string(decode(p, idx2answer), gt[i])
            for p, i in zip(preds, idxs)
        ]))

    base = honest_for(list(range(N)), 0.0)
    best = {}
    total = 0.0
    print(f"\n{'qtype':>12} | {'n':>4} | acc@0 | best | acc@best")
    for t in sorted(by_type, key=lambda k: -len(by_type[k])):
        idxs = by_type[t]
        a0 = honest_for(idxs, 0.0)
        if len(idxs) < 5:
            total += a0 * len(idxs)
            continue
        bb, ba = max(((b, honest_for(idxs, b)) for b in biases), key=lambda x: x[1])
        if bb != 0.0 and ba > a0:
            best[t] = bb
        total += ba * len(idxs)
        print(f"{t:>12} | {len(idxs):>4} | {a0:.3f} | {bb:>4} | {ba:.3f}")
    print(f"\n全体 honest: {base:.4f} → {total/N:.4f} (gain=+{total/N-base:.4f})")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="ckpt:backbone を並べる（例 a.pt:clip_vit_b_16 b.pt:clip_vit_l_14）")
    args = parser.parse_args()

    device = get_device()
    print("device =", device)
    os.makedirs("./outputs", exist_ok=True)

    # CLIP系は正規化・サイズ共通なので transform は1つでよい
    transform = build_transform()
    train_ds = VQADataset(df_path="./data/train_split.json",
                          image_dir="./data/train", transform=transform)
    valid_ds = VQADataset(df_path="./data/valid_split.json",
                          image_dir="./data/train", transform=transform)
    valid_ds.update_dict(train_ds)
    test_ds = VQADataset(df_path="./data/valid.json", image_dir="./data/valid",
                         transform=transform, answer=False)
    test_ds.update_dict(train_ds)

    idx2answer = train_ds.idx2answer
    unans = train_ds.answer2idx.get("unanswerable")
    dsets = {"valid": valid_ds, "test": test_ds}

    # 各モデルの確率を平均
    sum_valid = sum_test = None
    pairs = [m.rsplit(":", 1) for m in args.models]
    for ckpt, backbone in pairs:
        print(f"[model] {ckpt} ({backbone})")
        pr = model_probs(ckpt, backbone, train_ds, dsets, device)
        sum_valid = pr["valid"] if sum_valid is None else sum_valid + pr["valid"]
        sum_test = pr["test"] if sum_test is None else sum_test + pr["test"]
    n = len(pairs)
    avg_valid = sum_valid / n
    avg_test = sum_test / n
    print(f"\n{n} モデルを平均しました")

    # 平均確率の log を logits 扱いにして unanswerable バイアスを探索
    eps = 1e-9
    valid_logits = torch.log(avg_valid + eps)
    test_logits = torch.log(avg_test + eps)

    bias_by_qtype = {}
    if AUTO_SWEEP_UNANSWERABLE and unans is not None:
        bias_by_qtype = sweep_biases_on_logits(
            valid_logits, valid_ds, idx2answer, unans)
        print(f"採用 bias: {bias_by_qtype}")

    # test にバイアス適用 → デコード → 提出
    submission = []
    for i in range(test_logits.shape[0]):
        row = test_logits[i].clone()
        if unans is not None:
            qt = question_type(process_text(test_ds.df["question"][i]))
            row[unans] += bias_by_qtype.get(qt, 0.0)
        submission.append(decode(int(row.argmax()), idx2answer))

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    npy = f"./outputs/submission_ens_{ts}.npy"
    zip_ = f"./outputs/submission_ens_{ts}.zip"
    np.save(npy, np.array(submission))

    if AUTO_BUILD_NOTEBOOK:
        try:
            import runpy
            runpy.run_path("build_notebook.py", run_name="__main__")
        except Exception as e:
            print(f"[warn] notebook 生成失敗: {e}")
    with ZipFile(zip_, "w") as zf:
        zf.write(npy, arcname="submission.npy")
        if os.path.exists(MODEL_PATH):
            zf.write(MODEL_PATH, arcname="model.pt")
        if os.path.exists(NOTEBOOK_PATH):
            zf.write(NOTEBOOK_PATH, arcname=os.path.basename(NOTEBOOK_PATH))
    print(f"{zip_} created  (distinct={len(set(submission))})")


if __name__ == "__main__":
    main()

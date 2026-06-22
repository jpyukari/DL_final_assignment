"""
診断スクリプト: 「画像を一切使わず、質問だけで回答を予測する」モデルを学習し、
今の 0.5 のうちどれだけが質問理解（言語prior）で説明できるかを測る。

狙い（ボトルネック特定）:
  実験1  質問だけモデルの valid honest acc を見る。
         - 0.45 付近 → 今のスコアの大半は質問priorで決まっている＝画像はほぼ効いていない
         - 0.10 付近 → 質問priorは弱い＝画像（や融合）が本質的に重要
  実験2  質問タイプ別（color / how many / brand / yes-no / what ...）の acc。
         どのタイプで稼ぎ、どのタイプで落としているかを可視化。
  実験3  失敗例（honest acc < 0.5）をタイプ別に分類し、代表例をダンプ。

画像を読まないので軽量。GPU 不要（CPU でも数分）。

実行:
  python -m src.question_only
  python -m src.question_only --epochs 20 --lr 1e-3
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

from configs.baseline import MAX_QLEN, CLASS_WEIGHTS, SEED, BATCH_SIZE
from src.dataset import VQADataset, process_text, UNK_ANSWER
from src.metrics import vqa_acc_string
from src.utils import set_seed


# ---- 質問の意味タイプ分類（実験2/3 用）。先頭句ベースの粗い分類 ----
def question_type(q):
    """正規化済み質問文 q を粗い意味タイプに分類する。"""
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


def build_question_tensor(text, question2idx, max_qlen):
    """質問文 → 単語インデックス列 (max_qlen,)。dataset.__getitem__ と同じ規則。"""
    vocab = len(question2idx)
    unk_idx, pad_idx = vocab, vocab + 1
    ids = [
        question2idx.get(w, unk_idx)
        for w in process_text(text).split(" ") if w != ""
    ][:max_qlen]
    ids += [pad_idx] * (max_qlen - len(ids))
    return ids


def precompute(dataset, question2idx, answer2idx, with_target=True):
    """df から画像を読まずに (質問列, mode回答idx, 質問文) を作る。"""
    unk_ans = answer2idx[UNK_ANSWER]
    X, y, qtexts = [], [], []
    for i in range(len(dataset.df)):
        X.append(build_question_tensor(
            dataset.df["question"][i], question2idx, MAX_QLEN))
        qtexts.append(process_text(dataset.df["question"][i]))
        if with_target:
            ans = [
                answer2idx.get(process_text(a["answer"]), unk_ans)
                for a in dataset.df["answers"][i]
            ]
            y.append(Counter(ans).most_common(1)[0][0])  # mode
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long) if with_target else None
    return X, y, qtexts


class QuestionOnlyModel(nn.Module):
    """VQAモデルのテキスト分岐だけ（画像不使用）。
    encoder で語順の扱いを切替え、語順が効くかを測る:
      - "lstm": Embedding + 双方向LSTM の平均プール（語順あり）
      - "bow" : Embedding の平均プール（語順なし＝順序を捨てる）
    """

    def __init__(self, vocab_size, n_answer, encoder="lstm"):
        super().__init__()
        self.encoder = encoder
        self.pad_idx = vocab_size  # dataset の PAD = len(question2idx)+1 = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, 300, padding_idx=self.pad_idx)
        if encoder == "lstm":
            self.lstm = nn.LSTM(300, 256, batch_first=True, bidirectional=True)
            feat_dim = 512
        elif encoder == "bow":
            feat_dim = 300
        else:
            raise ValueError(f"unknown encoder: {encoder}")
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, n_answer),
        )

    def forward(self, q):
        mask = (q != self.pad_idx).unsqueeze(-1).float()
        emb = self.embedding(q)
        seq = self.lstm(emb)[0] if self.encoder == "lstm" else emb
        feat = (seq * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.fc(feat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder", choices=["lstm", "bow"], default="lstm",
                        help="lstm=語順あり / bow=語順なし（順序の効果測定用）")
    args = parser.parse_args()

    set_seed(SEED)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"  # 画像を使わない軽量診断なので CPU 許可
    )
    print("device =", device)

    # dict 構築用（init では画像を読まない）。transform は使わないので None。
    train_ds = VQADataset(df_path="./data/train_split.json",
                          image_dir="./data/train", transform=None)
    valid_ds = VQADataset(df_path="./data/valid_split.json",
                          image_dir="./data/train", transform=None)
    valid_ds.update_dict(train_ds)

    q2i = train_ds.question2idx
    a2i = train_ds.answer2idx
    idx2answer = train_ds.idx2answer
    vocab_size = len(q2i) + 1
    n_answer = len(a2i)
    print(f"vocab={len(q2i)}  n_answer={n_answer}")

    Xtr, ytr, _ = precompute(train_ds, q2i, a2i)
    Xva, yva, qva = precompute(valid_ds, q2i, a2i)
    print(f"train={len(Xtr)}  valid={len(Xva)}")

    # 質問の長さ統計（語順が効く余地があるかの目安）
    qlens = [len([w for w in q.split(" ") if w]) for q in qva]
    qlens = np.array(qlens)
    print(f"encoder={args.encoder}  質問語数: mean={qlens.mean():.1f} "
          f"median={int(np.median(qlens))} p90={int(np.percentile(qlens,90))} "
          f"max={qlens.max()}")

    # クラス重み（train と同条件で比較するため CLASS_WEIGHTS を反映）
    cw = torch.ones(n_answer)
    for lab, w in CLASS_WEIGHTS.items():
        if lab in a2i:
            cw[a2i[lab]] = w
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))

    model = QuestionOnlyModel(vocab_size, n_answer, encoder=args.encoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xva = Xva.to(device)
    N = len(Xtr)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        total = 0.0
        for s in range(0, N, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            logits = model(Xtr[idx])
            loss = criterion(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)

        # ---- valid 評価 ----
        model.eval()
        with torch.no_grad():
            preds = model(Xva).argmax(1).cpu().tolist()
        # honest acc（<unk>→unanswerable、GT は元文字列）
        honest = []
        for i, p in enumerate(preds):
            ps = idx2answer[p]
            if ps == UNK_ANSWER:
                ps = "unanswerable"
            gts = [process_text(a["answer"]) for a in valid_ds.df["answers"][i]]
            honest.append(vqa_acc_string(ps, gts))
        honest_acc = float(np.mean(honest))
        distinct = len(set(preds))
        print(f"Epoch [{ep+1}/{args.epochs}] train_loss={total/N:.4f} "
              f"valid honest acc={honest_acc:.4f} distinct_preds={distinct}")

    # ===== 実験2: 質問タイプ別 honest acc =====
    per_type = defaultdict(list)
    for q, a in zip(qva, honest):
        per_type[question_type(q)].append(a)
    print("\n-- 実験2: 質問タイプ別 honest acc（質問だけモデル） --")
    for t, v in sorted(per_type.items(), key=lambda kv: -len(kv[1])):
        print(f"  acc={np.mean(v):.3f}  n={len(v):4d}  {t}")

    # ===== 実験3: 失敗例（honest<0.5）をタイプ別に分類＋代表例 =====
    fail_by_type = defaultdict(list)
    for i, (q, a, p) in enumerate(zip(qva, honest, preds)):
        if a < 0.5:
            ps = idx2answer[p]
            ps = "unanswerable" if ps == UNK_ANSWER else ps
            gt = Counter(
                process_text(x["answer"]) for x in valid_ds.df["answers"][i]
            ).most_common(1)[0][0]
            fail_by_type[question_type(q)].append((q, ps, gt))
    print(f"\n-- 実験3: 失敗例(honest<0.5) のタイプ別件数 (総 valid={len(qva)}) --")
    for t, items in sorted(fail_by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(items):4d}  {t}")
    print("\n-- 失敗例サンプル（type: question | pred -> gt） --")
    for t, items in sorted(fail_by_type.items(), key=lambda kv: -len(kv[1])):
        for q, ps, gt in items[:3]:
            print(f"  [{t}] {q}  |  {ps} -> {gt}")


if __name__ == "__main__":
    main()

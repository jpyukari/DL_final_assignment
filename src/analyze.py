"""
施行ごとの「次に活かせるデータ」を集計するスクリプト。

- valid_split.json は GT（10人の回答）を持つので、ラベルごとの正答/誤答まで出せる
- 任意で submission の .npy を渡すと、テスト側の予測ラベル分布も出せる

出力:
  outputs/reports/report_<timestamp>.json   … 機械可読（次回の比較用）
  outputs/reports/report_<timestamp>.txt    … 人が読むサマリ

実行:
  python -m src.analyze
  python -m src.analyze --submission ./outputs/submission_xxxx.npy
"""
import os
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch

from configs.baseline import *

from src.dataset import VQADataset, process_text
from src.models.baseline import VQAModel
from src.utils import build_transform
from src.metrics import leaderboard_faithful_acc


def sample_vqa_acc(pred_idx, answer_idxs):
    """1 サンプルの VQA accuracy（10人投票, leave-one-out）。"""
    acc = 0.0
    n = len(answer_idxs)
    for i in range(n):
        num_match = sum(
            1 for j in range(n)
            if j != i and pred_idx == answer_idxs[j]
        )
        acc += min(num_match / 3, 1)
    return acc / n


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("GPU (CUDA/MPS) が利用できません。")


@torch.no_grad()
def collect_predictions(model, dataset, device):
    """valid_split 全件について (pred, mode, answers, question) を集める。"""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    preds, modes, answers_list = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        question = batch["question"].to(device)
        pred = model(image, question).argmax(1).cpu().numpy()
        preds.extend(pred.tolist())
        modes.extend(batch["mode_answer"].numpy().tolist())
        answers_list.extend(batch["answers"].int().numpy().tolist())

    questions = [process_text(q) for q in dataset.df["question"]]
    return preds, modes, answers_list, questions


def analyze(model_path, submission_npy=None, top_k=25, min_support=5, resnet=None):
    device = get_device()
    print("device =", device)

    # config の RESNET を CLI で上書き可能に（古い checkpoint の分析用）
    resnet = resnet or RESNET

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

    idx2answer = train_dataset.idx2answer

    model = VQAModel(
        vocab_size=len(train_dataset.question2idx) + 1,
        n_answer=len(train_dataset.answer2idx),
        backbone=resnet,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, modes, answers_list, questions = collect_predictions(
        model, valid_dataset, device
    )

    # --- 1. ラベル総数 ---
    n_answer = len(train_dataset.answer2idx)
    n_vocab = len(train_dataset.question2idx)

    # --- 2. 予測ラベルの頻度ランキング（valid） ---
    pred_counter = Counter(idx2answer[p] for p in preds)
    pred_ranking = [
        {"label": lab, "count": c, "ratio": round(c / len(preds), 4)}
        for lab, c in pred_counter.most_common(top_k)
    ]

    # --- 3. ラベルごとの正答率（GT の最頻値でグループ化） ---
    per_label_acc = defaultdict(list)   # true_label -> [vqa_acc, ...]
    per_sample_acc = []
    confusion = Counter()               # (true, pred) for misses
    for p, m, ans in zip(preds, modes, answers_list):
        a = sample_vqa_acc(p, ans)
        per_sample_acc.append(a)
        per_label_acc[idx2answer[m]].append(a)
        if p != m:
            confusion[(idx2answer[m], idx2answer[p])] += 1

    overall_vqa_acc = float(np.mean(per_sample_acc))

    # リーダーボード相当の honest acc（<unk>→unanswerable 変換＋元文字列照合）。
    # index 照合の overall_vqa_acc は <unk> 同士一致で過大評価されるため、
    # public と整合する値はこちら。
    faithful_acc = leaderboard_faithful_acc(preds, valid_dataset, idx2answer)

    label_stats = []
    for lab, accs in per_label_acc.items():
        label_stats.append({
            "label": lab,
            "support": len(accs),
            "vqa_acc": round(float(np.mean(accs)), 4),
        })
    # support が少ないラベルはノイズなので min_support で足切り
    eligible = [s for s in label_stats if s["support"] >= min_support]
    worst = sorted(eligible, key=lambda s: s["vqa_acc"])[:top_k]
    best = sorted(eligible, key=lambda s: -s["vqa_acc"])[:top_k]

    # --- 4. 混同しやすいペア（true -> pred の誤答） ---
    top_confusions = [
        {"true": t, "pred": p, "count": c}
        for (t, p), c in confusion.most_common(top_k)
    ]

    # --- 5. 質問タイプ別（質問の先頭語）精度 ---
    qtype_acc = defaultdict(list)
    for q, a in zip(questions, per_sample_acc):
        head = q.split(" ")[0] if q else "<empty>"
        qtype_acc[head].append(a)
    qtype_stats = sorted(
        [
            {"qtype": h, "count": len(v), "vqa_acc": round(float(np.mean(v)), 4)}
            for h, v in qtype_acc.items()
        ],
        key=lambda s: -s["count"],
    )[:top_k]

    # --- 6. 予測の偏り（特定ラベルへの集中度） ---
    used_labels = len(pred_counter)
    concentration = {
        lab: round(pred_counter.get(lab, 0) / len(preds), 4)
        for lab in ["yes", "no", "unanswerable"]
    }

    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "config": {
            "RESNET": resnet,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "LOSS_TYPE": LOSS_TYPE,
            "BATCH_SIZE": BATCH_SIZE,
            "NUM_EPOCHS": NUM_EPOCHS,
            "IMAGE_SIZE": IMAGE_SIZE,
            "FUSION": FUSION,
            "AUX_IMAGE_LOSS_WEIGHT": AUX_IMAGE_LOSS_WEIGHT,
            "MIN_ANSWER_COUNT": MIN_ANSWER_COUNT,
            "model_path": model_path,
        },
        "label_space": {
            "n_answer_labels": n_answer,
            "n_question_vocab": n_vocab,
        },
        "valid": {
            "n_samples": len(preds),
            "overall_vqa_acc": round(overall_vqa_acc, 4),
            "faithful_vqa_acc": round(faithful_acc, 4),
            "distinct_predicted_labels": used_labels,
            "concentration_yes_no_unanswerable": concentration,
            "prediction_ranking": pred_ranking,
            "worst_labels": worst,
            "best_labels": best,
            "top_confusions": top_confusions,
            "question_type_acc": qtype_stats,
        },
    }

    # --- 任意: submission(.npy) のラベル分布（テスト側、GT なし） ---
    if submission_npy and os.path.exists(submission_npy):
        sub = np.load(submission_npy, allow_pickle=True)
        sub_counter = Counter(sub.tolist())
        report["submission"] = {
            "path": submission_npy,
            "n_samples": int(len(sub)),
            "distinct_labels": len(sub_counter),
            "ranking": [
                {"label": lab, "count": c, "ratio": round(c / len(sub), 4)}
                for lab, c in sub_counter.most_common(top_k)
            ],
        }

    return report


def write_report(report):
    os.makedirs("./outputs/reports", exist_ok=True)
    ts = report["timestamp"]
    json_path = f"./outputs/reports/report_{ts}.json"
    txt_path = f"./outputs/reports/report_{ts}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    v = report["valid"]
    lines = []
    L = lines.append
    L(f"=== Report {ts} ===")
    L(f"config: {report['config']}")
    L(f"ラベル総数(answer): {report['label_space']['n_answer_labels']} / "
      f"質問語彙: {report['label_space']['n_question_vocab']}")
    L(f"valid: n={v['n_samples']}  VQA acc(index)={v['overall_vqa_acc']}  "
      f"VQA acc(LB相当/honest)={v['faithful_vqa_acc']}  ← publicと整合するのはこちら")
    L(f"予測に使われたラベル種類: {v['distinct_predicted_labels']}")
    L(f"yes/no/unanswerable 集中度: {v['concentration_yes_no_unanswerable']}")

    L("\n-- 予測ラベル頻度 TOP --")
    for r in v["prediction_ranking"]:
        L(f"  {r['count']:5d} ({r['ratio']:.2%})  {r['label']}")

    L("\n-- 外したラベル WORST (support>=min) --")
    for r in v["worst_labels"]:
        L(f"  acc={r['vqa_acc']:.3f} n={r['support']:4d}  {r['label']}")

    L("\n-- 正答が多いラベル BEST --")
    for r in v["best_labels"]:
        L(f"  acc={r['vqa_acc']:.3f} n={r['support']:4d}  {r['label']}")

    L("\n-- 混同 TOP (true -> pred) --")
    for r in v["top_confusions"]:
        L(f"  {r['count']:4d}  {r['true']}  ->  {r['pred']}")

    L("\n-- 質問タイプ別 acc --")
    for r in v["question_type_acc"]:
        L(f"  acc={r['vqa_acc']:.3f} n={r['count']:5d}  '{r['qtype']}'")

    if "submission" in report:
        s = report["submission"]
        L(f"\n-- submission 分布: {s['path']} (n={s['n_samples']}, "
          f"distinct={s['distinct_labels']}) --")
        for r in s["ranking"]:
            L(f"  {r['count']:5d} ({r['ratio']:.2%})  {r['label']}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nsaved: {json_path}\n       {txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--submission", default=None)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument(
        "--resnet", default=None,
        help="config の RESNET を上書き (例: resnet34)。古い checkpoint 用",
    )
    args = parser.parse_args()

    report = analyze(
        model_path=args.model,
        submission_npy=args.submission,
        top_k=args.top_k,
        min_support=args.min_support,
        resnet=args.resnet,
    )
    write_report(report)


if __name__ == "__main__":
    main()

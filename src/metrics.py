def VQA_criterion(batch_pred, batch_answers):
    """
    VQA タスクに用いられる評価関数．
    """
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


def vqa_acc_string(pred_str, gt_strings):
    """1サンプルの VQA accuracy（文字列ベース, leave-one-out）。"""
    acc = 0.0
    n = len(gt_strings)
    for i in range(n):
        num_match = sum(
            1 for j in range(n)
            if j != i and pred_str == gt_strings[j]
        )
        acc += min(num_match / 3, 1)
    return acc / n


def leaderboard_faithful_acc(preds, dataset, idx2answer):
    """リーダーボードと同じ採点での valid VQA accuracy。

    インデックス同士で照合する VQA_criterion は <unk> 同士が一致扱いになり
    valid を過大評価する（本番には <unk> ラベルが無いため）。これを避けるため:
      - 予測 idx → 文字列。<unk> は提出と同じく "unanswerable" に変換
      - GT は <unk> に潰さない「元の回答文字列」と照合
    こうすると valid が public とズレなくなる。

    Parameters
    ----------
    preds : list[int]
        dataset と同じ並び順の予測インデックス（valid は shuffle=False 前提）
    dataset : VQADataset
        元の回答文字列 dataset.df["answers"] を持つもの
    idx2answer : dict
    """
    from src.dataset import process_text, UNK_ANSWER

    total = 0.0
    for i, p in enumerate(preds):
        pred_str = idx2answer[p]
        if pred_str == UNK_ANSWER:
            pred_str = "unanswerable"
        gt_strings = [
            process_text(a["answer"]) for a in dataset.df["answers"][i]
        ]
        total += vqa_acc_string(pred_str, gt_strings)
    return total / len(preds)



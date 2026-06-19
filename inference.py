import os
import numpy as np
import torch

from zipfile import ZipFile
from datetime import datetime

from tqdm import tqdm

from configs.baseline import *

from src.dataset import VQADataset, UNK_ANSWER
from src.models.baseline import VQAModel
from src.utils import build_transform


timestamp = datetime.now().strftime("%Y%m%d_%H%M")

submission_npy = f"./outputs/submission_{timestamp}.npy"
submission_zip = f"./outputs/submission_{timestamp}.zip"


def main():

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        raise RuntimeError(
            "GPU (CUDA/MPS) が利用できません。"
            "CPU 実行は許可されていません。"
        )

    print("device =", device)

    os.makedirs("./outputs", exist_ok=True)

    transform = build_transform()

    print("loading train dataset...")

    train_dataset = VQADataset(
        df_path="./data/train_split.json",
        image_dir="./data/train",
        transform=transform,
    )

    print("loading valid dataset...")

    test_dataset = VQADataset(
        df_path="./data/valid.json",
        image_dir="./data/valid",
        transform=transform,
        answer=False,
    )

    test_dataset.update_dict(
        train_dataset
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("building model...")

    model = VQAModel(
        vocab_size=len(train_dataset.question2idx) + 1,
        n_answer=len(train_dataset.answer2idx),
        backbone=RESNET,
    ).to(device)

    print("loading checkpoint...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    unanswerable_idx = train_dataset.answer2idx.get("unanswerable")
    if UNANSWERABLE_LOGIT_BIAS and unanswerable_idx is not None:
        print(
            f"unanswerable logit を -{UNANSWERABLE_LOGIT_BIAS} 補正します "
            f"(idx={unanswerable_idx})"
        )

    submission = []

    print("start inference...")

    with torch.no_grad():

        for batch in tqdm(test_loader, desc="inference"):

            image = batch["image"].to(device)
            question = batch["question"].to(device)

            pred = model(
                image,
                question
            )

            if UNANSWERABLE_LOGIT_BIAS and unanswerable_idx is not None:
                pred[:, unanswerable_idx] -= UNANSWERABLE_LOGIT_BIAS

            pred = pred.argmax(1).item()

            answer = train_dataset.idx2answer[pred]

            # <unk>（足切りされた希少回答の受け皿）は提出すると 0 点なので、
            # 高頻度で部分点が入りやすい unanswerable に振り替える。
            if answer == UNK_ANSWER:
                answer = "unanswerable"

            submission.append(answer)

    submission = np.array(submission)

    np.save(
        submission_npy,
        submission
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"model weights not found: {MODEL_PATH} "
            f"(train.py を先に実行してください)"
        )

    if not os.path.exists(NOTEBOOK_PATH):
        raise FileNotFoundError(
            f"notebook not found: {NOTEBOOK_PATH} "
            f"(提出には統合Notebookが必須です)"
        )

    with ZipFile(
        submission_zip,
        "w"
    ) as zf:

        zf.write(
            submission_npy,
            arcname="submission.npy"
        )

        zf.write(
            MODEL_PATH,
            arcname="model.pt"
        )

        zf.write(
            NOTEBOOK_PATH,
            arcname=os.path.basename(NOTEBOOK_PATH),
        )

    print(f"{submission_zip} created")

    # 推論で使ったモデル + 生成した submission をそのまま分析してレポート保存
    print("running analysis...")
    try:
        from src.analyze import analyze, write_report
        report = analyze(
            model_path=MODEL_PATH,
            submission_npy=submission_npy,
        )
        write_report(report)
    except Exception as e:
        print(f"[warn] analyze をスキップしました: {e}")


if __name__ == "__main__":
    main()

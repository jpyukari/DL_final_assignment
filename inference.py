import os
import numpy as np
import torch

from zipfile import ZipFile
from datetime import datetime

from tqdm import tqdm

from configs.baseline import *

from src.dataset import VQADataset, UNK_ANSWER, process_text
from src.models.baseline import VQAModel
from src.metrics import decode_combined_pred
from src.utils import build_transform, apply_unanswerable_bias


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

    # 提出に使う unanswerable bias を決定。
    # AUTO_SWEEP_UNANSWERABLE: valid_split でタイプ別 best bias を自動探索して採用
    # （手動 sweep→config 貼り付けが不要）。TRAIN_ON_ALL 時は valid が学習に含まれ
    # 過学習になるので無効化し、config の値を使う。
    bias_by_qtype = dict(UNANSWERABLE_BIAS_BY_QTYPE)
    bias_default = UNANSWERABLE_BIAS_DEFAULT
    if AUTO_SWEEP_UNANSWERABLE and not TRAIN_ON_ALL and unanswerable_idx is not None:
        from src.sweep_unanswerable import sweep_biases
        print("auto-sweep: valid_split でタイプ別 unanswerable bias を探索...")
        valid_split = VQADataset(
            df_path="./data/valid_split.json",
            image_dir="./data/train",
            transform=transform,
        )
        valid_split.update_dict(train_dataset)
        bias_by_qtype, _, _, _ = sweep_biases(
            model, valid_split, train_dataset.idx2answer,
            unanswerable_idx, device, verbose=True,
        )
        bias_default = 0.0
        print(f"auto-sweep 採用 bias: {bias_by_qtype}")
    elif (bias_by_qtype or bias_default) and unanswerable_idx is not None:
        print(
            f"config の unanswerable bias を使用 "
            f"(default={bias_default}, by_qtype={bias_by_qtype})"
        )

    submission = []

    print("start inference...")

    sample_i = 0  # shuffle=False 前提。df の行に対応させて質問タイプを引く
    with torch.no_grad():

        for batch in tqdm(test_loader, desc="inference"):

            image = batch["image"].to(device)
            question = batch["question"].to(device)

            if OCR_ENABLED:
                pred = model(
                    image, question,
                    batch["ocr_char_ids"].to(device),
                    batch["ocr_mask"].to(device),
                )
            else:
                pred = model(image, question)

            # この batch に対応する質問文（タイプ別 unanswerable 補正用）
            bs = pred.shape[0]
            qtexts = [
                process_text(test_dataset.df["question"][sample_i + j])
                for j in range(bs)
            ]
            apply_unanswerable_bias(
                pred, qtexts, unanswerable_idx,
                bias_by_qtype, bias_default,
            )

            pred_idx = pred.argmax(1).item()

            # 結合出力をデコード（OCR位置ならコピートークン文字列、<unk>→unanswerable）。
            ocr_toks = (
                test_dataset.ocr_token_strings[sample_i]
                if OCR_ENABLED else None
            )
            answer = decode_combined_pred(
                pred_idx, train_dataset.idx2answer, ocr_toks)

            sample_i += bs

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

    # 提出zip には統合Notebookが必須。未ビルドでも走り切るよう自動生成する。
    if AUTO_BUILD_NOTEBOOK:
        print("building submission notebook...")
        try:
            import runpy
            runpy.run_path("build_notebook.py", run_name="__main__")
        except Exception as e:
            print(f"[warn] notebook 自動生成に失敗: {e}")

    if not os.path.exists(NOTEBOOK_PATH):
        raise FileNotFoundError(
            f"notebook not found: {NOTEBOOK_PATH} "
            f"(提出には統合Notebookが必須です。build_notebook.py を確認)"
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

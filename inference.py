import os
import numpy as np
import torch

from zipfile import ZipFile
from datetime import datetime

from tqdm import tqdm

from torchvision import transforms

from configs.baseline import *

from src.dataset import VQADataset
from src.models.baseline import VQAModel


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

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
    ])

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
    ).to(device)

    print("loading checkpoint...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

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

            pred = pred.argmax(1).item()

            submission.append(
                train_dataset.idx2answer[pred]
            )

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


if __name__ == "__main__":
    main()

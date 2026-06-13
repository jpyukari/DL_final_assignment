import os
import numpy as np
import torch

from zipfile import ZipFile
from datetime import datetime

from torchvision import transforms

from configs.baseline import *

from src.dataset import VQADataset
from src.models.baseline import VQAModel


timestamp = datetime.now().strftime("%Y%m%d_%H%M")

submission_npy = f"./outputs/submission_{timestamp}.npy"
submission_zip = f"./outputs/submission_{timestamp}.zip"


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs("./outputs", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
    ])

    train_dataset = VQADataset(
        df_path="./data/train.json",
        image_dir="./data/train",
        transform=transform,
    )

    test_dataset = VQADataset(
        df_path="./data/test.json",
        image_dir="./data/test",
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
    )

    model = VQAModel(
        vocab_size=len(train_dataset.question2idx) + 1,
        n_answer=len(train_dataset.answer2idx),
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    submission = []

    with torch.no_grad():

        for batch in test_loader:

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

    with ZipFile(
        submission_zip,
        "w"
    ) as zf:

        zf.write(
            submission_npy,
            arcname="submission.npy"
        )

        zf.write(
            "./outputs/checkpoints/best_model.pt",
            arcname="model.pt"
        )

        if os.path.exists(NOTEBOOK_PATH):
            zf.write(
                NOTEBOOK_PATH,
                arcname="submission.ipynb"
            )

    print(f"{submission_zip} created")


if __name__ == "__main__":
    main()

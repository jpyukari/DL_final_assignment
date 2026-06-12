import os
import time

import torch
import torch.nn as nn

from torchvision import transforms

from configs.baseline import *

from src.dataset import VQADataset
from src.metrics import VQA_criterion
from src.utils import set_seed

from src.models.baseline import VQAModel

from src.loss import (
    SoftCrossEntropyLoss,
    build_soft_target,
)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0
    total_acc = 0
    total_simple_acc = 0

    start = time.time()

    for batch in dataloader:

        image = batch["image"].to(device)
        question = batch["question"].to(device)
        answers = batch["answers"].to(device)
        mode_answer = batch["mode_answer"].to(device)

        pred = model(image, question)

        if LOSS_TYPE == "hard":

            loss = criterion(
                pred,
                mode_answer
            )

        elif LOSS_TYPE == "soft":

            soft_target = build_soft_target(
                answers,
                pred.shape[1]
            )

            loss = criterion(
                pred,
                soft_target
            )

        else:
            raise ValueError(
                f"Unknown LOSS_TYPE: {LOSS_TYPE}"
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        total_acc += VQA_criterion(
            pred.argmax(1),
            answers
        )

        total_simple_acc += (
            pred.argmax(1) == mode_answer
        ).float().mean().item()

    return (
        total_loss / len(dataloader),
        total_acc / len(dataloader),
        total_simple_acc / len(dataloader),
        time.time() - start,
    )


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0
    total_acc = 0
    total_simple_acc = 0

    start = time.time()

    for batch in dataloader:

        image = batch["image"].to(device)
        question = batch["question"].to(device)
        answers = batch["answers"].to(device)
        mode_answer = batch["mode_answer"].to(device)

        pred = model(image, question)

        if LOSS_TYPE == "hard":

            loss = criterion(
                pred,
                mode_answer
            )

        elif LOSS_TYPE == "soft":

            soft_target = build_soft_target(
                answers,
                pred.shape[1]
            )

            loss = criterion(
                pred,
                soft_target
            )

        total_loss += loss.item()

        total_acc += VQA_criterion(
            pred.argmax(1),
            answers
        )

        total_simple_acc += (
            pred.argmax(1) == mode_answer
        ).float().mean().item()

    return (
        total_loss / len(dataloader),
        total_acc / len(dataloader),
        total_simple_acc / len(dataloader),
        time.time() - start,
    )


def main():

    set_seed(SEED)

    os.makedirs("./outputs/checkpoints", exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    valid_dataset = VQADataset(
        df_path="./data/valid.json",
        image_dir="./data/valid",
        transform=transform,
    )

    valid_dataset.update_dict(
        train_dataset
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = VQAModel(
        vocab_size=len(train_dataset.question2idx) + 1,
        n_answer=len(train_dataset.answer2idx),
    ).to(device)

    if LOSS_TYPE == "hard":

        criterion = nn.CrossEntropyLoss()

    elif LOSS_TYPE == "soft":

        criterion = SoftCrossEntropyLoss()

    else:

        raise ValueError(
            f"Unknown LOSS_TYPE: {LOSS_TYPE}"
        )

    if OPTIMIZER == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

    else:

        raise ValueError(
            f"Unknown OPTIMIZER: {OPTIMIZER}"
        )

    best_acc = -1.0

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc, train_simple_acc, train_time = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
        )

        valid_loss, valid_acc, valid_simple_acc, valid_time = (
            validate(
                model,
                valid_loader,
                criterion,
                device,
            )
        )

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Acc={train_acc:.4f} "
            f"Valid Acc={valid_acc:.4f}"
        )

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(
                model.state_dict(),
                "./outputs/checkpoints/best_model.pt"
            )

    torch.save(
        model.state_dict(),
        "./outputs/checkpoints/model.pt"
    )


if __name__ == "__main__":
    main()
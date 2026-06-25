import os
import time
import math

import torch
import torch.nn as nn

from tqdm import tqdm

from configs.baseline import *

from src.dataset import VQADataset
from src.metrics import VQA_criterion, leaderboard_faithful_acc
from src.utils import set_seed, build_transform

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
    unanswerable_idx=None,
    desc="train",
):
    model.train()

    total_loss = 0
    total_acc = 0
    total_simple_acc = 0

    start = time.time()

    pbar = tqdm(dataloader, desc=desc, leave=True)

    for batch in pbar:

        image = batch["image"].to(device)
        question = batch["question"].to(device)
        answers = batch["answers"].to(device)
        mode_answer = batch["mode_answer"].to(device)

        # #3: AUX_IMAGE_LOSS_WEIGHT>0 のとき画像onlyの補助予測も受け取る
        use_aux = AUX_IMAGE_LOSS_WEIGHT > 0
        if use_aux:
            pred, aux = model(image, question, return_aux=True)
        else:
            pred = model(image, question)

        if LOSS_TYPE == "hard":

            loss = criterion(
                pred,
                mode_answer
            )
            if use_aux:
                loss = loss + AUX_IMAGE_LOSS_WEIGHT * criterion(aux, mode_answer)

        elif LOSS_TYPE == "soft":

            soft_target = build_soft_target(
                answers,
                pred.shape[1],
                ignore_index=unanswerable_idx,
            )

            loss = criterion(
                pred,
                soft_target
            )
            if use_aux:
                loss = loss + AUX_IMAGE_LOSS_WEIGHT * criterion(aux, soft_target)

        else:
            raise ValueError(
                f"Unknown LOSS_TYPE: {LOSS_TYPE}"
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"loss が NaN/Inf になりました (loss={loss.item()})。"
                "学習が発散しています。LR を下げる等の対策をしてください。"
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

        pbar.set_postfix(loss=f"{loss.item():.4f}")

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
    unanswerable_idx=None,
    desc="valid",
):
    model.eval()

    total_loss = 0
    total_acc = 0
    total_simple_acc = 0
    all_preds = []  # honest acc 用に予測 idx を順序通り収集

    start = time.time()

    for batch in tqdm(dataloader, desc=desc, leave=True):

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
                pred.shape[1],
                ignore_index=unanswerable_idx,
            )

            loss = criterion(
                pred,
                soft_target
            )

        total_loss += loss.item()

        all_preds.extend(pred.argmax(1).cpu().tolist())

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
        all_preds,
    )


def main():

    print("1")
    set_seed(SEED)

    print("2")
    os.makedirs("./outputs/checkpoints", exist_ok=True)

    print("3")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        raise RuntimeError(
            "GPU (CUDA/MPS) が利用できません。"
            "CPU 実行は許可されていません。"
        )

    print("4")

    # 学習は aug あり、推論/検証は aug なし（eval）。少数クラスには強い aug。
    train_transform = build_transform(train=True)
    strong_transform = build_transform(train=True, strong=True)
    eval_transform = build_transform(train=False)

    # TRAIN_ON_ALL: train_split + valid_split を結合して全データ学習（最終提出用）
    train_df_path = (
        ["./data/train_split.json", "./data/valid_split.json"]
        if TRAIN_ON_ALL else "./data/train_split.json"
    )

    train_dataset = VQADataset(

        df_path=train_df_path,

        image_dir="./data/train",

        transform=train_transform,

        strong_transform=strong_transform,

    )

    print("5")
    if TRAIN_ON_ALL:
        print("TRAIN_ON_ALL: train+valid 全データで学習（holdout/early-stopping なし）")
        valid_loader = None
        total_epochs = FINAL_EPOCHS
    else:
        valid_dataset = VQADataset(

            df_path="./data/valid_split.json",

            image_dir="./data/train",

            transform=eval_transform,

        )
        print("6")
        valid_dataset.update_dict(
            train_dataset
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        total_epochs = NUM_EPOCHS

    print("7")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    print("9")
    model = VQAModel(
        vocab_size=len(train_dataset.question2idx) + 1,
        n_answer=len(train_dataset.answer2idx),
        backbone=RESNET,
    )

    # 自己教師あり事前学習のバックボーン重みを初期値としてロード（スクラッチResNet用）。
    # ViT 利用時は model.resnet が無く、ImageNet 重みを直接ロード済みなので skip。
    if PRETRAINED_BACKBONE and hasattr(model, "resnet"):
        sd = torch.load(PRETRAINED_BACKBONE, map_location="cpu")
        missing, unexpected = model.resnet.load_state_dict(sd, strict=False)
        print(
            f"[SSL] backbone をロード: {PRETRAINED_BACKBONE} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif getattr(model, "use_vit", False):
        print("[backbone] ImageNet 事前学習 ViT-B/16 を fine-tune します")
    else:
        print("[SSL] PRETRAINED_BACKBONE 未設定 → スクラッチ学習")

    unanswerable_idx = (
        train_dataset.answer2idx.get("unanswerable")
        if EXCLUDE_UNANSWERABLE else None
    )
    print("9.5")

    model = model.to(device)
    print("10")
    print("11")

    # クラス重みベクトルを構築（頻出ラベルの loss を下げる）
    class_weights = torch.ones(len(train_dataset.answer2idx))
    applied = {}
    for label, w in CLASS_WEIGHTS.items():
        idx = train_dataset.answer2idx.get(label)
        if idx is not None:
            class_weights[idx] = w
            if w != 1.0:
                applied[label] = w
        else:
            print(f"[warn] CLASS_WEIGHTS のラベル '{label}' は語彙に無いので無視")
    if applied:
        print(f"[class_weights] 適用(≠1.0): {applied}")
    else:
        print("[class_weights] 全て 1.0 = 無効（CLASS_WEIGHTS を変えても学習は変わりません）")
    class_weights = class_weights.to(device)

    if LOSS_TYPE == "hard":

        criterion = nn.CrossEntropyLoss(weight=class_weights)

    elif LOSS_TYPE == "soft":

        criterion = SoftCrossEntropyLoss(weight=class_weights)

    else:

        raise ValueError(
            f"Unknown LOSS_TYPE: {LOSS_TYPE}"
        )

    # 過学習抑制: CLIP画像encoderの前段ブロックを凍結（小データFTの定番）。
    if FREEZE_BACKBONE_BLOCKS > 0 and getattr(model, "use_clip", False):
        # transformers のバージョン差に対応（.vision_model がある版／無い版）
        vm = getattr(model.clip, "vision_model", model.clip)
        for p in vm.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(vm.encoder.layers):
            if i < FREEZE_BACKBONE_BLOCKS:
                for p in layer.parameters():
                    p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"[freeze] CLIP前段 {FREEZE_BACKBONE_BLOCKS} ブロック凍結 "
              f"(凍結パラメータ {n_frozen/1e6:.0f}M)")

    # 事前学習バックボーン（vit.*/clip.*/cnn.*/resnet.*）は LR を下げて fine-tune し、
    # 新規ヘッド（LSTM/融合/fc 等）は通常 LR で学習する param group を作る。
    # 凍結した層（requires_grad=False）は optimizer に渡さない。
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("vit.", "clip.", "cnn.", "resnet.")):
            backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = [{"params": head_params, "lr": LR}]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": LR * BACKBONE_LR_MULT})
    print(
        f"[optim] head lr={LR}, backbone lr={LR * BACKBONE_LR_MULT} "
        f"(n_backbone={len(backbone_params)}, n_head={len(head_params)})"
    )

    if OPTIMIZER == "adam":

        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=WEIGHT_DECAY,
        )

    else:

        raise ValueError(
            f"Unknown OPTIMIZER: {OPTIMIZER}"
        )
    print("12")
    best_acc = -1.0
    epochs_no_improve = 0
    print("13")

    for epoch in range(total_epochs):

        train_loss, train_acc, train_simple_acc, train_time = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                unanswerable_idx=unanswerable_idx,
                desc=f"train [{epoch+1}/{total_epochs}]",
            )
        )

        # TRAIN_ON_ALL: valid が無いので validation/early-stopping をスキップし、
        # FINAL_EPOCHS まで回しきって best_model.pt を保存する。
        if TRAIN_ON_ALL:
            print(
                f"Epoch [{epoch+1}/{total_epochs}] "
                f"Train Acc={train_acc:.4f} (all-data)"
            )
            torch.save(
                model.state_dict(),
                "./outputs/checkpoints/best_model.pt"
            )
            continue

        valid_loss, valid_acc, valid_simple_acc, valid_time, valid_preds = (
            validate(
                model,
                valid_loader,
                criterion,
                device,
                unanswerable_idx=unanswerable_idx,
                desc=f"valid [{epoch+1}/{total_epochs}]",
            )
        )

        # honest acc（<unk>→unanswerable 変換＋元文字列照合）= public と整合する指標。
        # index 照合の valid_acc は <unk> 同士一致で過大評価されるので、
        # best モデル選択・early-stopping はこちらで行う。
        valid_faithful = leaderboard_faithful_acc(
            valid_preds, valid_dataset, train_dataset.idx2answer
        )

        # 崩壊（collapse）指標: valid で実際に予測された distinct ラベル数。
        # 小さい（数〜十数）ほど少数クラスへの崩壊が進んでいる＝悪い。
        distinct_preds = len(set(valid_preds))

        print(
            f"Epoch [{epoch+1}/{total_epochs}] "
            f"Train Acc={train_acc:.4f} "
            f"Valid Acc(index)={valid_acc:.4f} "
            f"Valid Acc(honest/LB)={valid_faithful:.4f} "
            f"distinct_preds={distinct_preds}"
        )

        if not math.isfinite(valid_loss):
            raise RuntimeError(
                f"valid_loss が NaN/Inf です (epoch {epoch+1})。"
                "発散したモデルを best として保存しないため停止します。"
            )

        if valid_faithful > best_acc:
            best_acc = valid_faithful
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                "./outputs/checkpoints/best_model.pt"
            )
        else:
            epochs_no_improve += 1
            print(
                f"No improvement for {epochs_no_improve}/{PATIENCE} "
                f"epoch(s) (best honest Valid Acc={best_acc:.4f})"
            )

            if epochs_no_improve >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(best honest Valid Acc={best_acc:.4f})"
                )
                break

    torch.save(
        model.state_dict(),
        "./outputs/checkpoints/model.pt"
    )

    # 学習完了後、そのまま推論→提出物生成→分析まで自動実行する。
    # （inference は MODEL_PATH=best_model.pt を読むので、保存済みの best を使う）
    print("=== train 完了。続けて inference を実行します ===")
    import inference
    inference.main()


if __name__ == "__main__":
    main()

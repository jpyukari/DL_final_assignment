"""
src/ の実装から提出用の統合 Notebook を生成するスクリプト。

出力: DL_Basic_2026_Spring_competition_VQA.ipynb
（必要なコードを全て 1 ファイルにまとめた自己完結 Notebook）
"""
import json
import re


def read_body(path, drop_import_substrings=()):
    """
    ソースファイルを読み、import 文を除いた本体を返す。
    drop_import_substrings に該当する import 行も併せて除去する。
    """
    lines = open(path, encoding="utf-8").read().splitlines()
    out = []
    for ln in lines:
        stripped = ln.strip()
        # トップレベル import はまとめてあるので除去
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        out.append(ln)
    # 連続する空行を整理（先頭・末尾の空行も削る）
    text = "\n".join(out).strip("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


cells = []

# ---- タイトル ----
cells.append(md(
    "# Deep Learning 基礎講座 最終課題: VQA\n"
    "\n"
    "画像と質問から回答を予測するタスク。\n"
    "本 Notebook は学習〜推論〜提出ファイル作成までを 1 ファイルにまとめた提出用ファイルです。\n"
    "\n"
    "実行すると以下を生成します。\n"
    "- `submission.npy` : テストデータ(valid.json)に対する予測\n"
    "- `model.pt` : 予測に使用したモデルの重み\n"
    "- `submission.zip` : 上記 + 本 Notebook をまとめた提出用 zip\n"
))

# ---- imports ----
cells.append(md("## 1. import"))
cells.append(code(
    "import os\n"
    "import re\n"
    "import time\n"
    "import random\n"
    "from statistics import mode\n"
    "from zipfile import ZipFile\n"
    "\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from PIL import Image\n"
    "\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torchvision import transforms\n"
))

# ---- config ----
cells.append(md("## 2. 設定 (config)\n\n実験設定はここを変更する。"))
cells.append(code(read_body("configs/baseline.py")))

# ---- utils ----
cells.append(md("## 3. utils"))
cells.append(code(read_body("src/utils.py")))

# ---- dataset ----
cells.append(md("## 4. データセット"))
cells.append(code(read_body("src/dataset.py")))

# ---- metrics ----
cells.append(md("## 5. 評価指標"))
cells.append(code(read_body("src/metrics.py")))

# ---- model: resnet ----
cells.append(md("## 6. モデル (ResNet)"))
cells.append(code(read_body("src/models/resnet.py")))

# ---- model: VQAModel ----
cells.append(md("## 7. モデル (VQAModel)"))
cells.append(code(read_body("src/models/baseline.py")))

# ---- loss ----
cells.append(md("## 8. 損失関数"))
cells.append(code(read_body("src/loss.py")))

# ---- train / validate ----
cells.append(md("## 9. 学習・検証ループ"))
cells.append(code(
    "def train_one_epoch(model, dataloader, optimizer, criterion, device, desc=\"train\"):\n"
    "    model.train()\n"
    "\n"
    "    total_loss = 0\n"
    "    total_acc = 0\n"
    "    total_simple_acc = 0\n"
    "\n"
    "    start = time.time()\n"
    "\n"
    "    pbar = tqdm(dataloader, desc=desc, leave=True)\n"
    "    for batch in pbar:\n"
    "        image = batch[\"image\"].to(device)\n"
    "        question = batch[\"question\"].to(device)\n"
    "        answers = batch[\"answers\"].to(device)\n"
    "        mode_answer = batch[\"mode_answer\"].to(device)\n"
    "\n"
    "        pred = model(image, question)\n"
    "\n"
    "        if LOSS_TYPE == \"hard\":\n"
    "            loss = criterion(pred, mode_answer)\n"
    "        elif LOSS_TYPE == \"soft\":\n"
    "            soft_target = build_soft_target(answers, pred.shape[1])\n"
    "            loss = criterion(pred, soft_target)\n"
    "        else:\n"
    "            raise ValueError(f\"Unknown LOSS_TYPE: {LOSS_TYPE}\")\n"
    "\n"
    "        optimizer.zero_grad()\n"
    "        loss.backward()\n"
    "        optimizer.step()\n"
    "\n"
    "        total_loss += loss.item()\n"
    "        total_acc += VQA_criterion(pred.argmax(1), answers)\n"
    "        total_simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()\n"
    "        pbar.set_postfix(loss=f\"{loss.item():.4f}\")\n"
    "\n"
    "    n = len(dataloader)\n"
    "    return total_loss / n, total_acc / n, total_simple_acc / n, time.time() - start\n"
    "\n"
    "\n"
    "@torch.no_grad()\n"
    "def validate(model, dataloader, criterion, device, desc=\"valid\"):\n"
    "    model.eval()\n"
    "\n"
    "    total_loss = 0\n"
    "    total_acc = 0\n"
    "    total_simple_acc = 0\n"
    "\n"
    "    start = time.time()\n"
    "\n"
    "    for batch in tqdm(dataloader, desc=desc, leave=True):\n"
    "        image = batch[\"image\"].to(device)\n"
    "        question = batch[\"question\"].to(device)\n"
    "        answers = batch[\"answers\"].to(device)\n"
    "        mode_answer = batch[\"mode_answer\"].to(device)\n"
    "\n"
    "        pred = model(image, question)\n"
    "\n"
    "        if LOSS_TYPE == \"hard\":\n"
    "            loss = criterion(pred, mode_answer)\n"
    "        elif LOSS_TYPE == \"soft\":\n"
    "            soft_target = build_soft_target(answers, pred.shape[1])\n"
    "            loss = criterion(pred, soft_target)\n"
    "\n"
    "        total_loss += loss.item()\n"
    "        total_acc += VQA_criterion(pred.argmax(1), answers)\n"
    "        total_simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()\n"
    "\n"
    "    n = len(dataloader)\n"
    "    return total_loss / n, total_acc / n, total_simple_acc / n, time.time() - start\n"
))

# ---- data preparation ----
cells.append(md(
    "## 10. データの準備\n\n"
    "学習データ・検証データ・テストデータを読み込む。"
    "辞書は学習データのものを検証・テストへ反映する。"
))
cells.append(code(
    "set_seed(SEED)\n"
    "\n"
    "if torch.cuda.is_available():\n"
    "    device = \"cuda\"\n"
    "elif torch.backends.mps.is_available():\n"
    "    device = \"mps\"\n"
    "else:\n"
    "    raise RuntimeError(\n"
    "        \"GPU (CUDA/MPS) が利用できません。CPU 実行は許可されていません。\"\n"
    "    )\n"
    "print(\"device =\", device)\n"
    "\n"
    "transform = transforms.Compose([\n"
    "    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),\n"
    "    transforms.ToTensor(),\n"
    "])\n"
    "\n"
    "train_dataset = VQADataset(\n"
    "    df_path=\"./data/train_split.json\",\n"
    "    image_dir=\"./data/train\",\n"
    "    transform=transform,\n"
    ")\n"
    "\n"
    "valid_dataset = VQADataset(\n"
    "    df_path=\"./data/valid_split.json\",\n"
    "    image_dir=\"./data/train\",\n"
    "    transform=transform,\n"
    ")\n"
    "valid_dataset.update_dict(train_dataset)\n"
    "\n"
    "train_loader = torch.utils.data.DataLoader(\n"
    "    train_dataset, batch_size=BATCH_SIZE, shuffle=True,\n"
    "    num_workers=2, pin_memory=True,\n"
    ")\n"
    "valid_loader = torch.utils.data.DataLoader(\n"
    "    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,\n"
    "    num_workers=2, pin_memory=True,\n"
    ")\n"
))

# ---- training ----
cells.append(md(
    "## 11. 学習\n\n"
    "検証 Acc が最良のモデルを `model.pt` として保存する。"
))
cells.append(code(
    "os.makedirs(\"./outputs/checkpoints\", exist_ok=True)\n"
    "\n"
    "model = VQAModel(\n"
    "    vocab_size=len(train_dataset.question2idx) + 1,\n"
    "    n_answer=len(train_dataset.answer2idx),\n"
    ").to(device)\n"
    "\n"
    "if LOSS_TYPE == \"hard\":\n"
    "    criterion = nn.CrossEntropyLoss()\n"
    "elif LOSS_TYPE == \"soft\":\n"
    "    criterion = SoftCrossEntropyLoss()\n"
    "else:\n"
    "    raise ValueError(f\"Unknown LOSS_TYPE: {LOSS_TYPE}\")\n"
    "\n"
    "if OPTIMIZER == \"adam\":\n"
    "    optimizer = torch.optim.Adam(\n"
    "        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,\n"
    "    )\n"
    "else:\n"
    "    raise ValueError(f\"Unknown OPTIMIZER: {OPTIMIZER}\")\n"
    "\n"
    "best_acc = -1.0\n"
    "\n"
    "for epoch in range(NUM_EPOCHS):\n"
    "    train_loss, train_acc, train_simple_acc, train_time = train_one_epoch(\n"
    "        model, train_loader, optimizer, criterion, device,\n"
    "        desc=f\"train [{epoch+1}/{NUM_EPOCHS}]\",\n"
    "    )\n"
    "    valid_loss, valid_acc, valid_simple_acc, valid_time = validate(\n"
    "        model, valid_loader, criterion, device,\n"
    "        desc=f\"valid [{epoch+1}/{NUM_EPOCHS}]\",\n"
    "    )\n"
    "    print(\n"
    "        f\"Epoch [{epoch+1}/{NUM_EPOCHS}] \"\n"
    "        f\"Train Acc={train_acc:.4f} Valid Acc={valid_acc:.4f}\"\n"
    "    )\n"
    "    if valid_acc > best_acc:\n"
    "        best_acc = valid_acc\n"
    "        torch.save(model.state_dict(), \"./model.pt\")\n"
    "\n"
    "print(\"best valid acc =\", best_acc)\n"
))

# ---- inference ----
cells.append(md(
    "## 12. 推論 (テストデータ)\n\n"
    "`data/valid.json` をテストデータとして予測し `submission.npy` を作成する。"
))
cells.append(code(
    "test_dataset = VQADataset(\n"
    "    df_path=\"./data/valid.json\",\n"
    "    image_dir=\"./data/valid\",\n"
    "    transform=transform,\n"
    "    answer=False,\n"
    ")\n"
    "test_dataset.update_dict(train_dataset)\n"
    "\n"
    "test_loader = torch.utils.data.DataLoader(\n"
    "    test_dataset, batch_size=1, shuffle=False,\n"
    "    num_workers=2, pin_memory=True,\n"
    ")\n"
    "\n"
    "model.load_state_dict(torch.load(\"./model.pt\", map_location=device))\n"
    "model.eval()\n"
    "\n"
    "submission = []\n"
    "with torch.no_grad():\n"
    "    for i, batch in enumerate(test_loader):\n"
    "        if i % 500 == 0:\n"
    "            print(f\"{i}/{len(test_dataset)}\")\n"
    "        image = batch[\"image\"].to(device)\n"
    "        question = batch[\"question\"].to(device)\n"
    "        pred = model(image, question)\n"
    "        pred = pred.argmax(1).item()\n"
    "        submission.append(train_dataset.idx2answer[pred])\n"
    "\n"
    "submission = np.array(submission)\n"
    "np.save(\"./submission.npy\", submission)\n"
    "print(\"submission.npy saved:\", submission.shape)\n"
))

# ---- submission zip ----
cells.append(md(
    "## 13. 提出ファイル作成\n\n"
    "`submission.npy` / `model.pt` / 本 Notebook をまとめて zip 化する。\n"
    "**注意**: `NOTEBOOK_NAME` を実際に保存している Notebook のファイル名に合わせること。"
))
cells.append(code(
    "# 本 Notebook のファイル名（保存名に合わせて変更）\n"
    "NOTEBOOK_NAME = \"DL_Basic_2026_Spring_competition_VQA.ipynb\"\n"
    "\n"
    "with ZipFile(\"submission.zip\", \"w\") as zf:\n"
    "    zf.write(\"submission.npy\")\n"
    "    zf.write(\"model.pt\")\n"
    "    if os.path.exists(NOTEBOOK_NAME):\n"
    "        zf.write(NOTEBOOK_NAME)\n"
    "    else:\n"
    "        print(f\"[警告] {NOTEBOOK_NAME} が見つかりません。\"\n"
    "              f\" Notebook を保存後にこのセルを再実行してください。\")\n"
    "\n"
    "print(\"submission.zip created\")\n"
))

# セル id を付与（nbformat の警告回避）
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "DL_Basic_2026_Spring_competition_VQA.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"wrote {out_path} ({len(cells)} cells)")

# VQA Project
https://github.com/jpyukari/DL_final_assignment.git

## omnicampus 環境セットアップ

chmod +x src/setup.sh

bash src/setup.sh

pip install scikit-learn

**##事前学習
**
# 1. SSL 事前学習（ラベル無し画像のみ）
python -m src.ssl_pretrain --epochs 100 --batch-size 256
#   → ./outputs/checkpoints/ssl_backbone.pt

# 2. configs/baseline.py を編集
PRETRAINED_BACKBONE = "./outputs/checkpoints/ssl_backbone.pt"

# 3. VQA 学習（バックボーンが初期化済みで始まる）
python train.py
#   ログに [SSL] backbone をロード ... missing=0 unexpected=0 が出れば成功


## 学習

bash python train.py 

学習済みモデルは

text outputs/checkpoints/best_model.pt 

に保存される。

---

## 推論・提出ファイル作成

bash python inference.py 

生成物

text outputs/submission.npy outputs/submission.zip 

---

## 実験設定

text configs/baseline.py 

を変更する。

変更対象

- MODEL
- LOSS_TYPE
- BATCH_SIZE
- LR
- NUM_EPOCHS

---

## モデル実装

text src/models/ 

新しいモデルを追加する場合はここに実装する。

---

## データ処理

text src/dataset.py 

---

## 評価指標

text src/metrics.py 

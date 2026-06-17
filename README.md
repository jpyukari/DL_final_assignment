# VQA Project

##Omnicampus環境セットアップ
chmod +x setup.sh
./setup.sh

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

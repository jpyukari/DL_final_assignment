SEED = 42

BATCH_SIZE = 128

NUM_EPOCHS = 25

PATIENCE = 3

# 最終提出用: True で train_split + valid_split を結合して全データ学習する。
# valid が無いので early-stopping は使えず、FINAL_EPOCHS 分だけ固定で回す。
# 運用: 開発(クリーンholdout)で early-stopping した epoch 数を FINAL_EPOCHS に入れ、
#       TRAIN_ON_ALL=True で再学習 → best_model.pt を提出に使う。
TRAIN_ON_ALL = False
FINAL_EPOCHS = 10

LR = 1e-4

WEIGHT_DECAY = 1e-5

OPTIMIZER = "adam"

LOSS_TYPE = "hard"

# True: 正解集計から "unanswerable" を除外する
#   - hard: mode_answer を unanswerable 以外で取る
#   - soft: soft target から unanswerable を除外する
# False: unanswerable も通常ラベルとして扱う
EXCLUDE_UNANSWERABLE = False

# クラス重み: 頻出ラベルの loss を下げて過適応（量産）を防ぐ。{ラベル: 重み}。
# 未指定ラベルは 1.0。空 dict {} で無効（全クラス 1.0）。hard/soft 両対応。
# 例: unanswerable/yes/no が多すぎるので学習時のペナルティを軽くする。
# <unk> は希少回答の寄せ集めで最大の「ゴミ箱クラス」。重みを下げて、
# 予測が <unk>/unanswerable の2クラスに崩壊するのを防ぐ。
CLASS_WEIGHTS = {
    "<unk>": 0.3,
    "unanswerable": 0.8,
    "yes": 3,
    "no": 3,
}

# 推論時に unanswerable の logit から引く値（学習はそのまま、推論だけ抑制）。
# 0.0 で無効。大きいほど unanswerable を出しにくくなる。
# 最適値は valid で選べる:  python -m src.sweep_unanswerable
UNANSWERABLE_LOGIT_BIAS = 0.0

# 回答語彙の最低出現回数。train でこの回数以上出た回答だけをクラスにする。
# それ未満の希少回答は "<unk>" にまとめる（1例しかないラベルは学習不能なため）。
# 1 で従来どおり全回答をクラス化。標準的な VQA は 8〜10 程度。
#   ≥1:40244  ≥3:7141  ≥5:4319  ≥8:2521  ≥10:1745 クラス
# min が大きいほど <unk> 行きが増える（mode が <unk>: min3=18.7% / min8=30.9%）
# 8→3 に下げて <unk> 吸い込みを 30.9%→18.7% に減らし、実ラベルを増やす
# （崩壊対策。クラス数 2521→7141 に増える）。
MIN_ANSWER_COUNT = 3

# 質問文の系列長。Embedding+LSTM のテキストエンコーダ用に、質問を
# 単語インデックス列としてこの長さに切り詰め／パディングする。
# （旧 bag-of-words 1層 Linear からの置き換え。語順を使えるようになる。）
MAX_QLEN = 20

IMAGE_SIZE = 360

# データ拡張（学習時のみ。inference/analyze には適用しない）。
# 色相(hue)は触らない: 色を答える質問が多く、hue揺らしは正解を壊すため。
AUGMENT = True
# 少数クラス対策: train 頻度がこの値以下のクラスのサンプルには、
# 通常の軽い aug ではなく「強い aug」を per-sample で適用して多様性を稼ぐ。
# 0 で無効（全データ一律の軽い aug のみ）。<unk> は対象外。
MINORITY_MAX_COUNT = 50

# 画像の正規化（ToTensor の後に適用）。train/inference/analyze で共通化される。
# True にする場合、学習・推論・分析の全てで同じ統計が使われる。
# mean/std は ImageNet 統計。スクラッチ学習なので、より厳密には
#   python -m src.compute_stats
# で train 画像から実測した値に差し替えるのが望ましい。
NORMALIZE = True
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

MODEL = "baseline"
EXP_NAME = "baseline"

# 画像だけ probe（src.color_probe）が対象にする質問タイプ。
#   "color" / "count" / "yes/no" / "what(other)" など（src.question_only の分類）。
# 画像だけで当てられるか＝画像に情報があるかを測る。config で対象を切替可能。
PROBE_QTYPE = "color"

# 画像と質問の融合方式（VQAModel が参照）。
#   "concat"          : 画像をavgpoolした512次元と質問特徴を連結（従来）。
#                       画像を1ベクトルに潰すので「どこを見るか」を質問で選べない。
#   "cross_attention" : 質問トークンが画像の空間特徴(H×W)に attention する。
#                       color/count など「画像の特定箇所を見る」問題向け。
# 診断（src.question_only）の結論: 質問priorは~0.59で飽和、伸びしろは画像側。
# cross_attention は画像情報を答えに効かせるための本命。
FUSION = "concat"

RESNET = "resnet50"  # 画像エンコーダ: "resnet18" / "resnet34" / "resnet50"

# 自己教師あり事前学習で得たバックボーン重みのパス。None でスクラッチ。
# 生成: python -m src.ssl_pretrain  → ./outputs/checkpoints/ssl_backbone.pt
# RESNET と同じ種別の重みであること（resnet50 で SSL したら resnet50 で使う）。
PRETRAINED_BACKBONE = PRETRAINED_BACKBONE = "./outputs/checkpoints/ssl_backbone.pt"


MODEL_PATH = "./outputs/checkpoints/best_model.pt"
NOTEBOOK_PATH = "./DL_Basic_2026_Spring_competition_VQA.ipynb"

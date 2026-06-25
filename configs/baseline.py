SEED = 42

BATCH_SIZE = 32   # ViT-L/14 は重いので縮小（OOMならさらに 16 へ。CLIP-B等なら128可）

NUM_EPOCHS = 25

PATIENCE = 3

# 最終提出用: True で train_split + valid_split を結合して全データ学習する。
# valid が無いので early-stopping は使えず、FINAL_EPOCHS 分だけ固定で回す。
# 運用: 開発(クリーンholdout)で early-stopping した epoch 数を FINAL_EPOCHS に入れ、
#       TRAIN_ON_ALL=True で再学習 → best_model.pt を提出に使う。
TRAIN_ON_ALL = False
FINAL_EPOCHS = 10

LR = 1e-4

# 事前学習バックボーン（ViT等）のfine-tune用に、backboneだけ LR を下げる倍率。
# 画像encoder（vit.*/resnet.*）の学習率 = LR × BACKBONE_LR_MULT。
# 事前学習特徴を壊さないため小さめ（0.05〜0.1）。1.0 で無効（全体同一LR）。
BACKBONE_LR_MULT = 0.05   # ViT-Lは大きく過学習しやすいので優しくFT（CLIP-Bは0.1で可）

WEIGHT_DECAY = 1e-5

OPTIMIZER = "adam"

# "hard": 最頻値1個を正解に CrossEntropy / "soft": 10人の回答のソフト分布で学習。
# soft は VQA の採点（複数回答の部分点）に整合するので honest が伸びやすい。
LOSS_TYPE = "soft"

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

# 推論時、質問タイプ別に unanswerable の logit に「足す」値（学習は不変・推論だけ調整）。
#   値 > 0: そのタイプで unanswerable を出しやすく（answerable率が低い count 等）
#   値 < 0: 出しにくく（answerable率が高い color 等）
# タイプ分類は src.utils.question_type（color/count/yes-no/what(other)/...）。
# 未指定タイプには UNANSWERABLE_BIAS_DEFAULT を使う。空 dict + default 0.0 で無効。
# 最適値は valid で自動探索:  python -m src.sweep_unanswerable
#   （タイプ別に honest acc を最大化する bias を出し、この dict 形式で表示する）
UNANSWERABLE_BIAS_BY_QTYPE = {
    # 例（sweep の結果を貼る）:
    # "count": 3.0,    # ほぼ unanswerable が正解
    # "color": -1.5,   # 画像で答えられるので unanswerable に逃がさない
}
UNANSWERABLE_BIAS_DEFAULT = 0.0

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

# 画像サイズ。ViT-B/16 は 224 固定（パッチ16×14×14=196）。ViT利用時は 224 必須。
IMAGE_SIZE = 224

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
FUSION = "cross_attention"

# #3「画像を捨てさせない学習」: 画像プール特徴だけから回答を予測する補助ヘッドを付け、
# 総ロス = 主ロス + AUX_IMAGE_LOSS_WEIGHT × 画像onlyロス。
# 画像branch に必ず勾配を流し、言語prior へのショートカット（画像無視）を防ぐ。
# 0.0 で無効。0.3〜1.0 程度から。推論には影響しない（補助ヘッドは推論で未使用）。
# 注意: train と inference で ≷0 を揃えること（アーキテクチャが変わるため）。
AUX_IMAGE_LOSS_WEIGHT = 0.0

# 画像エンコーダの種別（VQAModel が参照）。いずれも一般事前学習＋FT（合法）。
#   "clip_vit_b_16"     : CLIP ViT-B/16（画像-言語対照学習）。VQAに強い。本命の次段。
#                         transformers 必須。正規化は build_transform が自動でCLIP用に切替。
#   "vit_b_16"          : ImageNet ViT-B/16（86M, 224px固定）。
#   "efficientnet_v2_s" : EfficientNetV2-S（21M）。小さく過学習しにくい。
#   "efficientnet_b0"   : EfficientNet-B0（5M）。さらに軽量。
#   "resnet18/34/50"    : 自前スクラッチ ResNet（旧。~0.50 で頭打ち）。
# 注意: 切替時は train/inference で同じ値にすること（state_dict 整合）。
IMAGE_BACKBONE = "clip_vit_l_14"

RESNET = "resnet50"  # IMAGE_BACKBONE がスクラッチResNet時のみ有効

# 自己教師あり事前学習で得たバックボーン重みのパス。None でスクラッチ。
# 生成: python -m src.ssl_pretrain  → ./outputs/checkpoints/ssl_backbone.pt
# RESNET と同じ種別の重みであること（resnet50 で SSL したら resnet50 で使う）。
# SSL バックボーン（スクラッチResNet用）。ViT利用時は不要なので None。
# （ViT は IMAGE_BACKBONE で ImageNet 重みを直接ロードするため）
PRETRAINED_BACKBONE = None


# train.py を1回実行するだけで最後まで走らせるための自動化（inference.main が参照）。
# AUTO_SWEEP_UNANSWERABLE: 推論前に valid でタイプ別 unanswerable bias を自動探索し、
#   その結果を提出に適用する（手動 sweep→config 貼り付けが不要に）。
#   TRAIN_ON_ALL 時は valid が学習に含まれ過学習なので無効化される。
#   False のときは config の UNANSWERABLE_BIAS_BY_QTYPE をそのまま使う。
# AUTO_BUILD_NOTEBOOK: 提出zip の前に統合Notebookを自動生成（未ビルドでも走り切る）。
AUTO_SWEEP_UNANSWERABLE = True
AUTO_BUILD_NOTEBOOK = True

MODEL_PATH = "./outputs/checkpoints/best_model.pt"
NOTEBOOK_PATH = "./DL_Basic_2026_Spring_competition_VQA.ipynb"


SEED = 42

BATCH_SIZE = 64

NUM_EPOCHS = 15

PATIENCE = 3

LR = 1e-3

WEIGHT_DECAY = 1e-5

OPTIMIZER = "adam"

LOSS_TYPE = "hard"

# True: 正解集計から "unanswerable" を除外する
#   - hard: mode_answer を unanswerable 以外で取る
#   - soft: soft target から unanswerable を除外する
# False: unanswerable も通常ラベルとして扱う
EXCLUDE_UNANSWERABLE = False

# 推論時に unanswerable の logit から引く値（学習はそのまま、推論だけ抑制）。
# 0.0 で無効。大きいほど unanswerable を出しにくくなる。
# 最適値は valid で選べる:  python -m src.sweep_unanswerable
UNANSWERABLE_LOGIT_BIAS = 0.0

# 回答語彙の最低出現回数。train でこの回数以上出た回答だけをクラスにする。
# それ未満の希少回答は "<unk>" にまとめる（1例しかないラベルは学習不能なため）。
# 1 で従来どおり全回答をクラス化。標準的な VQA は 8〜10 程度。
#   ≥1:40244  ≥3:7141  ≥5:4319  ≥8:2521  ≥10:1745 クラス
# min が大きいほど <unk> 行きが増える（mode が <unk>: min3=18.7% / min8=30.9%）
MIN_ANSWER_COUNT = 3

IMAGE_SIZE = 224

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

RESNET = "resnet50"  # 画像エンコーダ: "resnet18" / "resnet34" / "resnet50"

MODEL_PATH = "./outputs/checkpoints/best_model.pt"
NOTEBOOK_PATH = "./DL_Basic_2026_Spring_competition_VQA.ipynb"

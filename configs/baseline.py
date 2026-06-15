SEED = 42

BATCH_SIZE = 64

NUM_EPOCHS = 15

PATIENCE = 3

LR = 1e-3

WEIGHT_DECAY = 1e-5

OPTIMIZER = "adam"

LOSS_TYPE = "hard"

IMAGE_SIZE = 224

MODEL = "baseline"
EXP_NAME = "baseline"

RESNET = "resnet34"  # 画像エンコーダ: "resnet18" / "resnet34" / "resnet50"

MODEL_PATH = "./outputs/checkpoints/best_model.pt"
NOTEBOOK_PATH = "./DL_Basic_2026_Spring_competition_VQA.ipynb"

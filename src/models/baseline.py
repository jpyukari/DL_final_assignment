import torch
import torch.nn as nn

from .resnet import ResNet18, ResNet34, ResNet50


# config の RESNET 文字列から ResNet 生成関数を引くためのテーブル
RESNET_FACTORY = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
}


class VQAModel(nn.Module):
    """
    VQA タスクを解くためのモデル例．
    """
    def __init__(self, vocab_size: int, n_answer: int, backbone: str = "resnet18"):
        """
        コンストラクタ．

        Parameters
        ----------
        vocab_size: int
            入力文の語彙数
        n_answer: int
            出力のクラス数
        backbone: str
            画像エンコーダの ResNet 種別 ("resnet18" / "resnet34" / "resnet50")
        """
        super().__init__()
        if backbone not in RESNET_FACTORY:
            raise ValueError(
                f"Unknown backbone: {backbone} "
                f"(choose from {list(RESNET_FACTORY)})"
            )
        self.resnet = RESNET_FACTORY[backbone]()  # いずれも 512 次元を出力
        self.text_encoder = nn.Linear(vocab_size, 512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):

        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(question)  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x
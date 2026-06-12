import torch
import torch.nn as nn

from .resnet import ResNet18


class VQAModel(nn.Module):
    """
    VQA タスクを解くためのモデル例．
    """
    def __init__(self, vocab_size: int, n_answer: int):
        """
        コンストラクタ．

        Parameters
        ----------
        vocab_size: int
            入力文の語彙数
        n_answer: int
            出力のクラス数
        """
        super().__init__()
        self.resnet = ResNet18()
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
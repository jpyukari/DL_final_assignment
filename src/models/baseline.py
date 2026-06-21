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

        # テキストエンコーダ: bag-of-words 1層 Linear から Embedding+双方向LSTM へ。
        # question は単語インデックス列 (B, MAX_QLEN)。語順を使えるので、
        # prior に逃げず質問内容に応じた回答を出しやすくなる（崩壊対策の本命）。
        # dataset 側: UNK=vocab_size-1, PAD=vocab_size（=ここでの pad_idx）。
        self.pad_idx = vocab_size  # dataset の PAD = len(question2idx)+1 = vocab_size
        emb_dim = 300
        hidden = 256  # 双方向なので text 特徴は 2*hidden = 512 次元
        self.embedding = nn.Embedding(
            vocab_size + 1, emb_dim, padding_idx=self.pad_idx
        )
        self.lstm = nn.LSTM(
            emb_dim, hidden, batch_first=True, bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):

        image_feature = self.resnet(image)  # 画像の特徴量 (B, 512)

        q = question.long()
        mask = (q != self.pad_idx).unsqueeze(-1).float()  # (B, L, 1) 実トークン=1
        emb = self.embedding(q)  # (B, L, emb_dim)
        lstm_out, _ = self.lstm(emb)  # (B, L, 2*hidden)
        # PAD を除いた時間方向の平均プーリング（全 PAD の保険で clamp）
        summed = (lstm_out * mask).sum(dim=1)  # (B, 2*hidden)
        counts = mask.sum(dim=1).clamp(min=1.0)
        question_feature = summed / counts  # (B, 512) テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x
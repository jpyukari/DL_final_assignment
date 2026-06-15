import math

import torch
import torch.nn as nn

from .resnet import ResNet18


class PositionalEncoding(nn.Module):
    """
    Transformer 用の正弦波位置エンコーディング．
    """
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1)]


class VQAModel(nn.Module):
    """
    VQA タスクを解くためのモデル．
    画像は ResNet18，質問は Transformer Encoder で符号化する．
    """
    def __init__(
        self,
        vocab_size: int,
        n_answer: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        """
        コンストラクタ．

        Parameters
        ----------
        vocab_size: int
            入力文の語彙数（未知語を含む）．パディング用 ID をこの後ろに 1 つ確保する．
        n_answer: int
            出力のクラス数
        """
        super().__init__()
        self.resnet = ResNet18()

        # vocab_size は未知語を含む語彙数．パディング用 ID をその直後に確保する．
        self.pad_idx = vocab_size
        self.embedding = nn.Embedding(
            vocab_size + 1, d_model, padding_idx=self.pad_idx
        )
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # enable_nested_tensor は MPS 未対応の nested tensor 最適化を使うため無効化する
        self.text_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.fc = nn.Sequential(
            nn.Linear(512 + d_model, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):

        image_feature = self.resnet(image)  # 画像の特徴量 (B, 512)

        question = question.long()
        pad_mask = question == self.pad_idx  # (B, L) True=パディング

        x = self.embedding(question)                          # (B, L, d_model)
        x = self.pos_encoder(x)
        x = self.text_encoder(x, src_key_padding_mask=pad_mask)  # (B, L, d_model)

        # パディングを除いた平均プーリングでテキスト特徴量を得る
        keep = (~pad_mask).unsqueeze(-1).float()  # (B, L, 1)
        question_feature = (x * keep).sum(1) / keep.sum(1).clamp(min=1e-6)

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x

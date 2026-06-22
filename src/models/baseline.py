import torch
import torch.nn as nn

from .resnet import ResNet18, ResNet34, ResNet50

from configs.baseline import FUSION, AUX_IMAGE_LOSS_WEIGHT


# config の RESNET 文字列から ResNet 生成関数を引くためのテーブル
RESNET_FACTORY = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
}


class VQAModel(nn.Module):
    """
    VQA タスクを解くためのモデル．

    融合方式は config の FUSION で切替（"concat" / "cross_attention"）。
    """
    def __init__(self, vocab_size: int, n_answer: int, backbone: str = "resnet18",
                 fusion: str = None):
        """
        Parameters
        ----------
        vocab_size: int
            入力文の語彙数（dataset が渡す len(question2idx)+1）
        n_answer: int
            出力のクラス数
        backbone: str
            画像エンコーダの ResNet 種別 ("resnet18" / "resnet34" / "resnet50")
        fusion: str
            融合方式。None なら config の FUSION を使う。
        """
        super().__init__()
        if backbone not in RESNET_FACTORY:
            raise ValueError(
                f"Unknown backbone: {backbone} "
                f"(choose from {list(RESNET_FACTORY)})"
            )
        self.fusion = fusion or FUSION
        if self.fusion not in ("concat", "cross_attention"):
            raise ValueError(f"Unknown FUSION: {self.fusion}")

        self.resnet = RESNET_FACTORY[backbone]()  # avgpool+fc 経由で 512 次元

        # テキストエンコーダ: Embedding + 双方向LSTM。
        # question は単語インデックス列 (B, MAX_QLEN)。
        # dataset 側: UNK=vocab_size-1, PAD=vocab_size（=ここでの pad_idx）。
        self.pad_idx = vocab_size
        emb_dim = 300
        hidden = 256  # 双方向なので text 特徴は 2*hidden = 512 次元
        self.d = 512
        self.embedding = nn.Embedding(
            vocab_size + 1, emb_dim, padding_idx=self.pad_idx
        )
        self.lstm = nn.LSTM(
            emb_dim, hidden, batch_first=True, bidirectional=True
        )

        if self.fusion == "cross_attention":
            # 画像の空間特徴 (B, C, H, W) を d 次元トークン列に射影し、
            # 質問トークン(query)が画像トークン(key/value)に attention する。
            self.img_proj = nn.Conv2d(self.resnet.out_channels, self.d, kernel_size=1)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.d, num_heads=8, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(self.d)

        # concat / cross_attention とも最終特徴は [質問512, 画像由来512] の連結。
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

        # 補助ヘッド（#3「画像を捨てさせない学習」）。画像プール特徴(512)だけから
        # 回答を予測する。AUX_IMAGE_LOSS_WEIGHT>0 のときだけ作る。学習で
        # 画像branch に必ず勾配を流し、言語prior へのショートカットを防ぐ。
        # 推論では使わない（forward の return_aux=False で無視）。
        # 注意: train と inference で AUX_IMAGE_LOSS_WEIGHT の ≷0 を揃えること
        #       （FUSION と同様、アーキテクチャが変わるため。state_dict 整合）。
        self.aux_image_fc = (
            nn.Linear(self.d, n_answer) if AUX_IMAGE_LOSS_WEIGHT > 0 else None
        )

    def _encode_question(self, question):
        """質問 → (トークン系列 (B,L,512), マスク (B,L,1), プール特徴 (B,512))。"""
        q = question.long()
        mask = (q != self.pad_idx).unsqueeze(-1).float()  # (B, L, 1)
        lstm_out, _ = self.lstm(self.embedding(q))  # (B, L, 512)
        pooled = (lstm_out * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # (B, 512)
        return lstm_out, mask, pooled

    def forward(self, image, question, return_aux=False):
        q_tokens, q_mask, q_pooled = self._encode_question(question)

        # backbone は1回だけ通し、空間特徴とプールベクトルの両方を得る
        # （concat/cross_attention/補助ヘッドで共有。二重forwardを避ける）。
        feat_map = self.resnet.forward_features(image)         # (B, C, H, W)
        pooled = self.resnet.avgpool(feat_map).flatten(1)      # (B, C)
        img_vec = self.resnet.fc(pooled)                       # (B, 512) = resnet(image)

        if self.fusion == "concat":
            # 画像を1ベクトル(512)に潰して質問プール特徴と連結（従来方式）。
            x = torch.cat([img_vec, q_pooled], dim=1)

        else:  # cross_attention
            feat = self.img_proj(feat_map)  # (B, d, H, W)
            img_tokens = feat.flatten(2).transpose(1, 2)  # (B, HW, d)

            # 質問トークン(query)が画像トークン(key/value)に attention。
            # 画像トークンは全て有効なので key_padding_mask は不要。
            # PAD 質問位置は後段のプールで mask により除外する。
            attended, _ = self.cross_attn(
                query=q_tokens, key=img_tokens, value=img_tokens,
            )
            attended = self.attn_norm(attended)  # (B, L, d)
            # PAD 質問位置を除いて平均 → 画像に接地した質問特徴 (B, d)
            grounded = (attended * q_mask).sum(1) / q_mask.sum(1).clamp(min=1.0)
            x = torch.cat([grounded, q_pooled], dim=1)

        logits = self.fc(x)
        if return_aux:
            # 画像プール特徴だけからの予測（#3 補助ロス用）。
            aux = self.aux_image_fc(img_vec) if self.aux_image_fc is not None else None
            return logits, aux
        return logits

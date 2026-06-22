import torch
import torch.nn as nn

from torchvision.models import vit_b_16, ViT_B_16_Weights

from .resnet import ResNet18, ResNet34, ResNet50

from configs.baseline import FUSION, AUX_IMAGE_LOSS_WEIGHT, IMAGE_BACKBONE


# config の RESNET 文字列から（レガシーな）スクラッチ ResNet 生成関数を引く
RESNET_FACTORY = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
}


class VQAModel(nn.Module):
    """
    VQA タスクを解くためのモデル．

    画像エンコーダは config の IMAGE_BACKBONE で切替:
      - "vit_b_16"          : ImageNet 事前学習 ViT-B/16（torchvision）を fine-tune。
                              ルール上「構成要素としての事前学習モデル＋FT」で許可。
      - "resnet18/34/50"    : 自前のスクラッチ ResNet（旧）。
    融合方式は config の FUSION で切替（"concat" / "cross_attention"）。
    """
    def __init__(self, vocab_size: int, n_answer: int, backbone: str = "resnet18",
                 fusion: str = None):
        super().__init__()
        self.fusion = fusion or FUSION
        if self.fusion not in ("concat", "cross_attention"):
            raise ValueError(f"Unknown FUSION: {self.fusion}")

        self.d = 512  # 融合空間の次元（質問512に揃える）

        # ---- 画像エンコーダ ----
        self.use_vit = (IMAGE_BACKBONE == "vit_b_16")
        if self.use_vit:
            # ImageNet 事前学習 ViT-B/16。heads を外して特徴抽出器として使う。
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.vit.heads = nn.Identity()
            img_dim = 768  # ViT-B hidden size（CLS/パッチtrークン次元）
            self.img_pool_proj = nn.Linear(img_dim, self.d)  # CLS → 512
            if self.fusion == "cross_attention":
                self.img_token_proj = nn.Linear(img_dim, self.d)  # patch → 512
        else:
            if backbone not in RESNET_FACTORY:
                raise ValueError(
                    f"Unknown backbone: {backbone} (choose {list(RESNET_FACTORY)})")
            self.resnet = RESNET_FACTORY[backbone]()  # avgpool+fc 経由で 512 次元
            if self.fusion == "cross_attention":
                self.img_proj = nn.Conv2d(
                    self.resnet.out_channels, self.d, kernel_size=1)

        # ---- テキストエンコーダ: Embedding + 双方向LSTM（質問側は飽和済みだが据え置き）。
        # question は単語インデックス列 (B, MAX_QLEN)。
        # dataset 側: UNK=vocab_size-1, PAD=vocab_size（=ここでの pad_idx）。
        self.pad_idx = vocab_size
        emb_dim = 300
        hidden = 256  # 双方向なので text 特徴は 2*hidden = 512 次元
        self.embedding = nn.Embedding(
            vocab_size + 1, emb_dim, padding_idx=self.pad_idx
        )
        self.lstm = nn.LSTM(
            emb_dim, hidden, batch_first=True, bidirectional=True
        )

        if self.fusion == "cross_attention":
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
        # 回答を予測する。AUX_IMAGE_LOSS_WEIGHT>0 のときだけ作る。
        # 注意: train と inference で AUX_IMAGE_LOSS_WEIGHT の ≷0、および
        #       IMAGE_BACKBONE / FUSION を揃えること（state_dict 整合のため）。
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

    def _vit_tokens(self, x):
        """ViT のトークン列 (B, 1+N, 768) を得る。[:,0]=CLS, [:,1:]=パッチ。"""
        x = self.vit._process_input(x)            # (B, N, 768) パッチ埋め込み
        n = x.shape[0]
        cls = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, 1+N, 768)
        return self.vit.encoder(x)                # pos埋め込み+Transformer+LN

    def _image_features(self, image):
        """画像 → (img_vec (B,512), img_tokens (B,N,512) or None)。
        img_tokens は cross_attention のときだけ作る。"""
        if self.use_vit:
            tokens = self._vit_tokens(image)               # (B, 1+N, 768)
            img_vec = self.img_pool_proj(tokens[:, 0])     # (B, 512) CLS
            img_tokens = (
                self.img_token_proj(tokens[:, 1:])         # (B, N, 512)
                if self.fusion == "cross_attention" else None
            )
        else:
            feat_map = self.resnet.forward_features(image)     # (B, C, H, W)
            pooled = self.resnet.avgpool(feat_map).flatten(1)  # (B, C)
            img_vec = self.resnet.fc(pooled)                   # (B, 512)
            img_tokens = (
                self.img_proj(feat_map).flatten(2).transpose(1, 2)  # (B, HW, 512)
                if self.fusion == "cross_attention" else None
            )
        return img_vec, img_tokens

    def forward(self, image, question, return_aux=False):
        q_tokens, q_mask, q_pooled = self._encode_question(question)
        img_vec, img_tokens = self._image_features(image)

        if self.fusion == "concat":
            x = torch.cat([img_vec, q_pooled], dim=1)
        else:  # cross_attention
            # 質問トークン(query)が画像トークン(key/value)に attention。
            # 画像トークンは全て有効なので key_padding_mask は不要。
            attended, _ = self.cross_attn(
                query=q_tokens, key=img_tokens, value=img_tokens,
            )
            attended = self.attn_norm(attended)  # (B, L, d)
            grounded = (attended * q_mask).sum(1) / q_mask.sum(1).clamp(min=1.0)
            x = torch.cat([grounded, q_pooled], dim=1)

        logits = self.fc(x)
        if return_aux:
            aux = self.aux_image_fc(img_vec) if self.aux_image_fc is not None else None
            return logits, aux
        return logits

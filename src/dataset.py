import re

import numpy as np
from PIL import Image

import pandas as pd
import torch

from torchvision import transforms

from statistics import mode
from collections import Counter

from configs.baseline import (
    EXCLUDE_UNANSWERABLE, MIN_ANSWER_COUNT, MINORITY_MAX_COUNT, MAX_QLEN,
)

UNK_ANSWER = "<unk>"


def process_text(text):
    """
    入力文と回答のフォーマットを統一するための関数．

    Parameters
    ----------
    text : str
        入力文，もしくは回答．
    """
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class VQADataset(torch.utils.data.Dataset):
    """
    VQA データセットを扱うためのクラス．
    """
    def __init__(self, df_path, image_dir, transform=None, answer=True,
                 strong_transform=None):
        self.transform = transform  # 画像の前処理（軽い aug もしくは eval）
        self.strong_transform = strong_transform  # 少数クラス向けの強い aug（任意）
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        # df_path は単一パスでも、複数パスのリスト（全データ学習用に train+valid 結合）でも可
        if isinstance(df_path, (list, tuple)):
            self.df = pd.concat(
                [pd.read_json(p) for p in df_path],
                ignore_index=True,
            )
        else:
            self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}
        self.minority_answer_idx = set()  # 強い aug 対象（answer=True 時に構築）

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答の出現回数を数え、MIN_ANSWER_COUNT 以上のものだけをクラス化する。
            # 希少回答（1例しかない等）は学習不能なので "<unk>" にまとめる。
            answer_counter = Counter()
            for answers in self.df["answers"]:
                for answer in answers:
                    answer_counter[process_text(answer["answer"])] += 1

            for word, count in answer_counter.items():
                if count >= MIN_ANSWER_COUNT:
                    self.answer2idx[word] = len(self.answer2idx)

            # OOV（足切りされた希少回答）受け皿
            self.answer2idx[UNK_ANSWER] = len(self.answer2idx)

            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

            # 少数クラス集合（強い aug 対象）。
            # 語彙に残った（count>=MIN_ANSWER_COUNT）かつ低頻度（<=MINORITY_MAX_COUNT）のクラス。
            # <unk> は多数の希少回答の集約なので対象外。
            self.minority_answer_idx = {
                self.answer2idx[word]
                for word, count in answer_counter.items()
                if word in self.answer2idx and count <= MINORITY_MAX_COUNT
            }

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (MAX_QLEN,) long
            質問文を単語インデックス列にしたもの（Embedding+LSTM 用）。
            未知語は UNK=len(question2idx)、空き要素は PAD=len(question2idx)+1。
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        # 画像は mode_answer（少数クラス判定）を求めてから transform を選ぶため、
        # ここでは PIL のまま読み込んでおく。
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}").convert("RGB")
        # 質問を単語インデックス列にする（語順を保持。Embedding+LSTM 用）。
        # UNK = 未知語の受け皿 = V、PAD = 空き要素 = V+1（V=語彙数）。
        vocab = len(self.question2idx)
        unk_idx = vocab
        pad_idx = vocab + 1
        question_words = process_text(self.df["question"][idx]).split(" ")
        q_ids = [
            self.question2idx.get(word, unk_idx)
            for word in question_words
            if word != ""
        ][:MAX_QLEN]
        q_ids += [pad_idx] * (MAX_QLEN - len(q_ids))  # 末尾を PAD で埋める
        question = torch.tensor(q_ids, dtype=torch.long)

        if self.answer:
            answer_texts = [
                process_text(answer["answer"])
                for answer in self.df["answers"][idx]
            ]

            # 足切りされた回答は <unk> に写像
            unk_idx = self.answer2idx[UNK_ANSWER]
            answers = [
                self.answer2idx.get(answer_text, unk_idx)
                for answer_text in answer_texts
            ]

            if EXCLUDE_UNANSWERABLE:
                # unanswerable を除外して最頻値を取る（全件 unanswerable の時はフォールバック）
                filtered_answers = [
                    self.answer2idx.get(answer_text, unk_idx)
                    for answer_text in answer_texts
                    if answer_text != "unanswerable"
                ]
                mode_answer_idx = mode(filtered_answers) if filtered_answers else mode(answers)
            else:
                mode_answer_idx = mode(answers)

            # 少数クラスのサンプルだけ強い aug を使う（strong_transform がある時のみ）
            if (
                self.strong_transform is not None
                and mode_answer_idx in self.minority_answer_idx
            ):
                image = self.strong_transform(image)
            else:
                image = self.transform(image)

            return {"image": image, "question": question,"answers": torch.Tensor(answers), "mode_answer": int(mode_answer_idx),}
        else:
            image = self.transform(image)
            return {"image": image, "question": question}

    def __len__(self):
        return len(self.df)



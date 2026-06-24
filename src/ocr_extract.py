"""
画像内テキストを EasyOCR で抽出し、画像ファイル名→トークン列の辞書を JSON 保存する。

OCR コピー機構（M4C-lite）の前処理。学習・推論のたびに OCR を回すと遅いので、
ここで一度だけ全画像を処理してキャッシュする。

ルール: OCR は「事前学習モデルを特徴抽出の構成要素として利用」する範囲。
配布データで OCR を学習し直すわけではない（汎用テキスト認識器の流用）。

生成物: data/ocr_tokens.json = {画像ファイル名: ["token", ...]}
使い方:
  pip install easyocr
  python -m src.ocr_extract                       # data/train, data/valid を処理
  python -m src.ocr_extract --dirs ./data/train   # 一部のみ
  （途中再開可: 既に JSON にあるファイルはスキップ）
"""
import os
import re
import glob
import json
import argparse

IMAGE_EXT = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

# OCR 結果の正規化: 小文字化し、英数と一部記号のみ残して単語分割。
_KEEP = re.compile(r"[^a-z0-9 ]+")


def normalize_tokens(text):
    """OCR の生テキスト → トークン列（小文字・記号除去・空白分割）。"""
    text = text.lower()
    text = _KEEP.sub(" ", text)
    return [t for t in text.split() if t]


def gather_images(dirs):
    paths = []
    for d in dirs:
        for ext in IMAGE_EXT:
            paths.extend(glob.glob(os.path.join(d, ext)))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+",
                        default=["./data/train", "./data/valid"])
    parser.add_argument("--out", default="./data/ocr_tokens.json")
    parser.add_argument("--max-tokens", type=int, default=40,
                        help="1画像あたり保存するトークン上限（重複除去後）")
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    import easyocr  # 遅延import（未使用環境で依存不要）
    import torch
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    # 途中再開: 既存 JSON を読み込み、未処理のみ回す
    cache = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"既存キャッシュ {len(cache)} 件を読み込み（再開）")

    paths = gather_images(args.dirs)
    todo = [p for p in paths if os.path.basename(p) not in cache]
    print(f"画像 {len(paths)} 件中 {len(todo)} 件を処理（dirs={args.dirs}）")

    from tqdm import tqdm
    for i, p in enumerate(tqdm(todo, desc="ocr")):
        fname = os.path.basename(p)
        try:
            lines = reader.readtext(p, detail=0)  # list[str]
        except Exception as e:
            print(f"[warn] {fname}: {e}")
            lines = []
        toks, seen = [], set()
        for line in lines:
            for t in normalize_tokens(line):
                if t not in seen:
                    seen.add(t)
                    toks.append(t)
        cache[fname] = toks[:args.max_tokens]

        if (i + 1) % args.save_every == 0:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            tqdm.write(f"  saved {len(cache)} 件")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    n_tok = sum(len(v) for v in cache.values())
    print(f"完了: {len(cache)} 画像, 平均 {n_tok/max(len(cache),1):.1f} トークン/画像")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

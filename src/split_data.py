import pandas as pd
from sklearn.model_selection import train_test_split
import re

def process_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = pd.read_json("./data/train.json")

train_df, valid_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
)

# train側の回答語彙を作成
train_answers = set()

for answers in train_df["answers"]:
    for a in answers:
        train_answers.add(
            process_text(a["answer"])
        )

# valid側で未知回答を含むサンプルをtrainへ移動
keep_valid_idx = []
move_to_train_idx = []

for idx, row in valid_df.iterrows():

    unknown = False

    for a in row["answers"]:
        ans = process_text(a["answer"])

        if ans not in train_answers:
            unknown = True
            break

    if unknown:
        move_to_train_idx.append(idx)
    else:
        keep_valid_idx.append(idx)

# 再構築
valid_df = valid_df.loc[keep_valid_idx]
extra_train = df.loc[move_to_train_idx]

train_df = pd.concat(
    [train_df, extra_train],
    ignore_index=True
)

train_df.to_json(
    "./data/train_split.json",
    orient="records",
)

valid_df.to_json(
    "./data/valid_split.json",
    orient="records",
)

print(
    f"train={len(train_df)}, "
    f"valid={len(valid_df)}"
)

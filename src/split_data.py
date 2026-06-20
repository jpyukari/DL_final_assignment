import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("./data/train.json")

# クリーンなランダム分割。
# 以前は「valid の回答が train 語彙に無いサンプルを train へ移動」していたが、
# それは valid を「全回答が既知＝簡単な問題」だけに絞ってしまい、
# valid acc が本番(LB)より不当に高くなる原因だった。
# 現在は dataset 側が未知回答を <unk> に写像する（KeyError しない）ため、
# この移動処理は不要。valid は本番と同じ分布のまま残す。
train_df, valid_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

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

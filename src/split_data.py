import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("./data/train.json")

train_df, valid_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
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

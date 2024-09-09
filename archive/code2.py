import numpy as np
import pandas as pd

# ================================================================ #

train: pd.DataFrame = pd.read_csv(
    "data/train.csv",
    na_values=[''],
    keep_default_na=False,
    index_col=0,
).convert_dtypes()

# ================================================================ #

print(pd.concat([
    train.isna().sum().to_frame(name="count"),
    train.dtypes.map(str).to_frame(name="type"),
], axis="columns").query("count > 0"))
print()
print(train.dtypes.map(str).value_counts().to_frame(name="count"))

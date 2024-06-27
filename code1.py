import numpy as np
import pandas as pd

# ================================================================ #

train: pd.DataFrame = pd.read_csv(
    "input/train.csv",
    na_values=[],
    keep_default_na=False,
    index_col=0,
).convert_dtypes()

# ================================================================ #

names = [name for name in train.columns if "NA" in train[name].tolist()]

print(len(names))
print(names)

correct = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

print(len(correct))
print(correct)

incorrect = [name for name in names if name not in correct]

print(len(incorrect))
print(incorrect)

print(pd.Series({
    name: (train[name] == "NA").sum()
    for name in incorrect
}))

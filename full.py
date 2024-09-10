# Import:

from _imports import *

# Config:

sklearn.set_config(transform_output='pandas')

# Load Data:

train_df: pd.DataFrame = pd.read_csv(
    'input/train.csv',
    na_values=[''],
    keep_default_na=False,
    index_col=0,
)
test_df: pd.DataFrame = pd.read_csv(
    'input/test.csv',
    na_values=[''],
    keep_default_na=False,
    index_col=0,
)

# Extract Data:

train_ids: np.ndarray = train_df.index.to_numpy()
train_X: pd.DataFrame = train_df.drop(columns=['SalePrice']).reset_index(drop=True)
train_y: np.ndarray = train_df['SalePrice'].to_numpy()
test_ids: np.ndarray = test_df.index.to_numpy()
test_X: pd.DataFrame = test_df.reset_index(drop=True)

# Names:

features: list[str] = train_X.columns.tolist()

# Types:

type_features: dict[str, list[str]] = {
    'nominal': [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature', 'MoSold',
        'SaleType', 'SaleCondition',
    ],
    'ordinal': [
        'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional',
        'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
    ],
    'numerical': [
        'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'YrSold',
    ],
}

assert np.array_equal([f for f in features if f in type_features['nominal']], type_features['nominal'])
assert np.array_equal([f for f in features if f in type_features['ordinal']], type_features['ordinal'])
assert np.array_equal([f for f in features if f in type_features['numerical']], type_features['numerical'])

feature_types: dict[str, str] = {
    feature: 'nominal' if feature in type_features['nominal'] else
    'ordinal' if feature in type_features['ordinal'] else
    'numerical' if feature in type_features['numerical'] else
    str(1 / 0)
    for feature in features
}

assert np.array_equal(list(feature_types.keys()), features)

# Cats and Levels:

discrete_values: dict[str, list[Any]] = {
    'MSSubClass': [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
    'MSZoning': ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
    'Street': ['Grvl', 'Pave'],
    'Alley': ['Grvl', 'Pave', 'NA'],
    'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
    'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
    'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
    'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
    'LandSlope': ['Gtl', 'Mod', 'Sev'],
    'Neighborhood': [
        'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR',
        'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer',
        'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker',
    ],
    'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
    'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
    'BldgType': ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'],
    'HouseStyle': ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
    'OverallQual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'OverallCond': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
    'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
    'Exterior1st': [
        'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other',
        'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing',
    ],
    'Exterior2nd': [
        'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other',
        'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing',
    ],
    'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'],
    'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
    'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'NA'],
    'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
    'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'CentralAir': ['N', 'Y'],
    'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
    'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
    'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'],
    'GarageFinish': ['Fin', 'RFn', 'Unf', 'NA'],
    'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'PavedDrive': ['Y', 'P', 'N'],
    'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
    'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'],
    'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'],
    'MoSold': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
    'SaleCondition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'],
}

assert np.array_equal([f for f in features if f in discrete_values.keys()], list(discrete_values.keys()))
assert np.array_equal([f for f, t in feature_types.items() if t != 'numerical'], list(discrete_values.keys()))

# Fix Typos:

typo_mapping_base: dict[str, dict[Any, Any]] = {
    'MSZoning': {
        'C (all)': 'C',
    },
    'BldgType': {
        '2fmCon': '2FmCon',
        'Duplex': 'Duplx',
        'Twnhs': 'TwnhsI',
    },
    'Exterior2nd': {
        'Brk Cmn': 'BrkComm',
        'CmentBd': 'CemntBd',
        'Wd Shng': 'WdShing',
    },
}

typo_mapping: dict[str, dict[Any, Any]] = {
    f: (typo_mapping_base.get(f) or {}) |
       ({} if (t != 'numerical' and 'NA' in discrete_values[f]) else {'NA': np.nan})
    for f, t in feature_types.items()
}


def fix_typos(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(typo_mapping)


typo_fixer = FunctionTransformer(fix_typos)


def check_cats(df: pd.DataFrame) -> None:
    non_numeric_features = [f for f, t in feature_types.items() if t != 'numerical']
    for f in non_numeric_features:
        assert set(df[f].dropna().unique()) <= set(discrete_values[f])


check_cats(train_X.replace(typo_mapping))
check_cats(test_X.replace(typo_mapping))


# Fix Dtypes:

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    for f in type_features['numerical']:
        out_df[f] = out_df[f].map(int, na_action='ignore')
    return out_df


dtype_fixer = FunctionTransformer(fix_dtypes)


# Fix Some of Masonry Veneer NAs:

def fix_masonry_veneer(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    out_df.loc[
        out_df['MasVnrArea'] == 0,
        'MasVnrType'
    ] = 'None'
    return out_df


masonry_veneer_fixer = FunctionTransformer(fix_masonry_veneer)


# Fix Remod Times:

def fix_remod_times(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    indices = out_df['YrSold'] < out_df['YearRemodAdd']
    out_df.loc[indices, 'YearRemodAdd'] = out_df.loc[indices, 'YearBuilt']
    return out_df


remod_times_fixer = FunctionTransformer(fix_remod_times)

# Fix Discrete Types:

discrete_int_features: list[str] = [
    k
    for k, v in discrete_values.items()
    if type(v[0]) is not str
]

discrete_strs: dict[str, list[str]] = {
    k: [f'C{vi}' for vi in v] if k in discrete_int_features else v
    for k, v in discrete_values.items()
}


def fix_discrete_types(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    for f in discrete_int_features:
        out_df[f] = out_df[f].map('C{}'.format, na_action='ignore')
    return out_df


discrete_type_fixer = FunctionTransformer(fix_discrete_types)

# Fix NA Names:

discrete_na_name_mapping: dict[str, dict[str, str]] = {
    f: {
        'NA': 'CNA',
        'None': 'CNone',
    }
    for f in discrete_values.keys()
}

discrete_na: dict[str, list[str]] = {
    k: ['CNA' if vi == 'NA' else 'CNone' if vi == 'None' else vi for vi in v]
    for k, v in discrete_strs.items()
}


def fix_na_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(discrete_na_name_mapping)


na_names_fixer = FunctionTransformer(fix_na_names)

# Fix dots:

discrete: dict[str, list[str]] = discrete_na | {
    'HouseStyle': ['1Story', '1d5Fin', '1d5Unf', '2Story', '2d5Fin', '2d5Unf', 'SFoyer', 'SLvl'],
}

dot_mapping: dict[str, dict[str, str]] = {
    'HouseStyle': {
        '1.5Fin': '1d5Fin',
        '1.5Unf': '1d5Unf',
        '2.5Fin': '2d5Fin',
        '2.5Unf': '2d5Unf',
    }
}


def fix_dots(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(dot_mapping)


dot_fixer = FunctionTransformer(fix_dots)

# Fixer Pipeline:

fixer = Pipeline([
    ('typo_fixer', typo_fixer),
    ('dtype_fixer', dtype_fixer),
    ('masonry_veneer_fixer', masonry_veneer_fixer),
    ('remod_times_fixer', remod_times_fixer),
    ('discrete_type_fixer', discrete_type_fixer),
    ('na_names_fixer', na_names_fixer),
    ('dot_fixer', dot_fixer),
], verbose=False)

# Impute:

imputer = ColumnTransformer(
    transformers=[
        ('numerical', SimpleImputer(strategy='median'), type_features['numerical']),
        ('ordinal', SimpleImputer(strategy='most_frequent'), type_features['ordinal']),
        ('nominal', SimpleImputer(strategy='most_frequent'), type_features['nominal']),
    ],
    remainder='drop',
    verbose=False,
    verbose_feature_names_out=False,
)

# Encode:

encoder = ColumnTransformer(
    transformers=[
        ('numerical', 'passthrough', type_features['numerical']),
        ('ordinal', OneHotEncoder(
            categories=[
                discrete[f]
                for f in type_features['ordinal']
            ],
            drop='first',
            sparse_output=False,
            handle_unknown='error',
        ), type_features['ordinal']),
        ('nominal', OneHotEncoder(
            categories=[
                discrete[f]
                for f in type_features['nominal']
            ],
            drop='first',
            sparse_output=False,
            handle_unknown='error',
        ), type_features['nominal']),
    ],
    remainder='drop',
    verbose=False,
    verbose_feature_names_out=False,
)

# Standardizer:

standardizer = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), type_features['numerical']),
    ],
    remainder='passthrough',
    verbose=False,
    verbose_feature_names_out=False,
)

# Pipeline:

pipeline = Pipeline([
    ('fixer', fixer),
    ('imputer', imputer),
    ('encoder', encoder),
    ('standardizer', standardizer),
], verbose=False)

# Do Fixing:

train_X_fixed: pd.DataFrame = pipeline.fit_transform(train_X)
test_X_fixed: pd.DataFrame = pipeline.transform(test_X)

# Save Data:

train_X_fixed.to_csv(
    'tmp/train_X.csv',
    na_rep='',
    index=False,
    float_format='%g',
)
test_X_fixed.to_csv(
    'tmp/test_X.csv',
    na_rep='',
    index=False,
    float_format='%g',
)
pd.Series(
    train_y,
    name='SalePrice',
).to_csv(
    'tmp/train_y.csv',
    na_rep='',
    index=False,
    float_format='%g',
)

# Save Some Meta Data:

pd.Series(
    index=pd.Index(discrete.keys(), name='feature'),
    data=[
        '|'.join(vs)
        for vs in discrete.values()
    ],
    name='values',
).to_csv(
    'tmp/meta/discrete.csv',
    na_rep='',
    index=True,
    float_format='%g',
)

pd.Series(
    index=pd.Index(feature_types.keys(), name='feature'),
    data=feature_types.values(),
    name='type',
).to_csv(
    'tmp/meta/types.csv',
    na_rep='',
    index=True,
    float_format='%g',
)

# Show Some Stats:

print(train_X_fixed.head())
print(train_X_fixed.describe().T)

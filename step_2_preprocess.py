# Import:

from _imports import *

sklearn.set_config(transform_output='pandas')

Path('data/preprocess/').mkdir(parents=True, exist_ok=True)
Path('data/preprocess/meta/').mkdir(parents=True, exist_ok=True)
Path('data/preprocess/model/').mkdir(parents=True, exist_ok=True)
Path('data/preprocess/na/').mkdir(parents=True, exist_ok=True)

# Load Meta Data:

types: dict[str, str] = pd.read_csv('data/fix/meta/types.csv', index_col=0).iloc[:, 0].to_dict()
discrete: dict[str, list[str]] = pd.read_csv('data/fix/meta/discrete.csv', index_col=0).iloc[:, 0] \
    .map(lambda x: x.split(sep='|')).to_dict()

type_features = {
    'numerical': [f for f, t in types.items() if t == 'numerical'],
    'ordinal': [f for f, t in types.items() if t == 'ordinal'],
    'nominal': [f for f, t in types.items() if t == 'nominal'],
}

# Load Data:

train_X: pd.DataFrame = pd.read_csv('data/fix/train_X.csv')
train_y: np.ndarray = pd.read_csv('data/fix/train_y.csv').iloc[:, 0].to_numpy()
test_X: pd.DataFrame = pd.read_csv('data/fix/test_X.csv')

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
        ('ordinal_ord', OrdinalEncoder(
            categories=[
                discrete[f]
                for f in type_features['ordinal']
            ],
            handle_unknown='error',
        ), type_features['ordinal']),
        ('ordinal_oh', OneHotEncoder(
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
        ('ordinal_ord', StandardScaler(), type_features['ordinal']),
    ],
    remainder='passthrough',
    verbose=False,
    verbose_feature_names_out=False,
)

# Preprocess:

preprocess = Pipeline([
    ('imputer', imputer),
    ('encoder', encoder),
    ('standardizer', standardizer),
], verbose=False)

# Do Processing:

train_X_prep: pd.DataFrame = preprocess.fit_transform(train_X, train_y).astype(np.float64)
test_X_prep: pd.DataFrame = preprocess.transform(test_X).astype(np.float64)

# Preprocess Target:

sklearn.set_config(transform_output='default')

target_pipeline = Pipeline([
    ('transformer', FunctionTransformer(
        func=np.log,
        inverse_func=np.exp,
        validate=True,
        accept_sparse=False,
        check_inverse=True,
        feature_names_out='one-to-one',
    )),
    ('standardizer', StandardScaler()),
], verbose=False)

train_y_prep: np.ndarray = target_pipeline.fit_transform(train_y.reshape(-1, 1))[:, 0]

target_pipeline_meta = {
    'transformer_forward': 'numpy.log',
    'transformer_backward': 'numpy.exp',
    'standardizer_loc': target_pipeline.named_steps['standardizer'].mean_[0],
    'standardizer_scale': target_pipeline.named_steps['standardizer'].scale_[0],
}


# Find NA Positions:

def base_feature(f: str) -> str:
    if "_" in f:
        return f[:f.index("_")]
    else:
        return f


train_X_nas: pd.DataFrame = pd.read_csv('data/fix/na/train_X.csv').eq(1)
test_X_nas: pd.DataFrame = pd.read_csv('data/fix/na/test_X.csv').eq(1)

features: list[str] = train_X.columns.tolist()
features_prep: list[str] = train_X_prep.columns.tolist()

train_X_prep_nas: pd.DataFrame = pd.DataFrame({
    f: train_X_nas.loc[:, base_feature(f)]
    for f in features_prep
})

test_X_prep_nas: pd.DataFrame = pd.DataFrame({
    f: test_X_nas.loc[:, base_feature(f)]
    for f in features_prep
})

# Save Data, Metadata and Models:

train_X_prep.to_csv(
    'data/preprocess/train_X.csv',
    index=False,
)

test_X_prep.to_csv(
    'data/preprocess/test_X.csv',
    index=False,
)

pd.Series(
    data=train_y_prep,
    name='SalePrice',
).to_csv(
    'data/preprocess/train_y.csv',
    index=False,
)

pd.Series(target_pipeline_meta).to_csv(
    'data/preprocess/meta/target.csv',
    header=False,
)

joblib.dump(
    preprocess,
    filename='data/preprocess/model/pipeline.pkl',
    protocol=pickle.HIGHEST_PROTOCOL,
    compress=True,
)

joblib.dump(
    target_pipeline,
    filename='data/preprocess/model/target_pipeline.pkl',
    protocol=pickle.HIGHEST_PROTOCOL,
    compress=True,
)

train_X_prep_nas.astype(int).to_csv(
    'data/preprocess/na/train_X.csv',
    index=False,
)

test_X_prep_nas.astype(int).to_csv(
    'data/preprocess/na/test_X.csv',
    index=False,
)

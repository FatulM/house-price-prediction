# Import:

from _imports import *

# Load Meta Data:

types: dict[str, str] = pd.read_csv('data/meta/types.csv', index_col=0).iloc[:, 0].to_dict()
discrete: dict[str, list[str]] = pd.read_csv('data/meta/discrete.csv', index_col=0).iloc[:, 0] \
    .map(lambda x: x.split(sep="|")).to_dict()

type_features = {
    "numerical": [f for f, t in types.items() if t == "numerical"],
    "ordinal": [f for f, t in types.items() if t == "ordinal"],
    "nominal": [f for f, t in types.items() if t == "nominal"],
}

# Load Data:

train_X: pd.DataFrame = pd.read_csv('data/train_X.csv')
train_y: np.ndarray = pd.read_csv('data/train_y.csv').iloc[:, 0].to_numpy()
test_X: pd.DataFrame = pd.read_csv('data/test_X.csv')

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

# Preprocess:

preprocess = Pipeline([
    ('imputer', imputer),
    ('encoder', encoder),
    ('standardizer', standardizer),
], verbose=False)

# Do Processing:

train_X_prep = preprocess.fit_transform(train_X, train_y).astype(np.float64)
test_X_prep = preprocess.transform(test_X).astype(np.float64)

# Save Results:

train_X_prep.to_csv('data/train_X_prep.csv', index=False)
test_X_prep.to_csv('data/test_X_prep.csv', index=False)

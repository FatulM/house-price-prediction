# Import:

from _imports import *

import keras

Path('data/pca/').mkdir(parents=True, exist_ok=True)
Path('data/pca/transform/').mkdir(parents=True, exist_ok=True)

# Load Data:

features: list[str] = pd.read_csv('data/preprocess/train_X.csv', nrows=10).columns.tolist()

train_X: np.ndarray = pd.read_csv('data/preprocess/train_X.csv').to_numpy()
test_X: np.ndarray = pd.read_csv('data/preprocess/test_X.csv').to_numpy()

train_X_nas: np.ndarray = pd.read_csv('data/preprocess/na/train_X.csv').eq(1).to_numpy()
test_X_nas: np.ndarray = pd.read_csv('data/preprocess/na/test_X.csv').eq(1).to_numpy()

train_y: np.ndarray = pd.read_csv('data/preprocess/train_y.csv').iloc[:, 0].to_numpy()


# Do PCA Imputation:

# noinspection PyPep8Naming
def pca_impute(
        X: np.ndarray, Xna: np.ndarray,
        Xt: np.ndarray, Xtna: np.ndarray,
        ncomp: int,
        niter: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    costs = np.zeros((niter, 4))
    Xf = X.copy()
    Xtf = Xt.copy()
    for i in range(niter):
        pca = PCA(n_components=ncomp, random_state=i)

        pca.fit(Xf)

        Z = pca.inverse_transform(pca.transform(Xf))
        Zt = pca.inverse_transform(pca.transform(Xtf))

        nXf = Xf.copy()
        nXf[Xna] = Z[Xna]
        nXtf = Xtf.copy()
        nXtf[Xtna] = Zt[Xtna]

        diff0 = math.sqrt(((nXf[Xna] - X[Xna]) ** 2).mean())
        diff = math.sqrt(((nXf[Xna] - Xf[Xna]) ** 2).mean())
        difft0 = math.sqrt(((nXtf[Xtna] - Xt[Xtna]) ** 2).mean())
        difft = math.sqrt(((nXtf[Xtna] - Xtf[Xtna]) ** 2).mean())

        costs[i, 0] = diff0
        costs[i, 1] = diff
        costs[i, 2] = difft0
        costs[i, 3] = difft

        Xf = nXf
        Xtf = nXtf
    return Xf, Xtf, costs


train_X_fixed, test_X_fixed, _ = pca_impute(
    train_X, train_X_nas,
    test_X, test_X_nas,
    2,
    100,
)

# Save Data:

pd.DataFrame(
    columns=features,
    data=train_X_fixed,
).to_csv(
    'data/pca/train_X.csv',
    index=False,
    float_format="%.15f",
)

pd.DataFrame(
    columns=features,
    data=test_X_fixed,
).to_csv(
    'data/pca/test_X.csv',
    index=False,
    float_format="%.15f",
)

# Predict:

keras.utils.set_random_seed(100)

model = keras.saving.load_model("data/predict/model/model.keras")

train_y_pred = model.predict(train_X_fixed)[:, 0]
test_y_pred = model.predict(test_X_fixed)[:, 0]

# Transform:

target_transformer = joblib.load('data/preprocess/model/target_pipeline.pkl')

train_y_pred_t = target_transformer.inverse_transform(train_y_pred.reshape(-1, 1))[:, 0]
test_y_pred_t = target_transformer.inverse_transform(test_y_pred.reshape(-1, 1))[:, 0]

# Save Data:

pd.Series(
    data=train_y_pred,
    name='SalePrice',
).to_csv(
    'data/pca/train_y.csv',
    index=False,
    float_format="%.15f",
)

pd.Series(
    data=test_y_pred,
    name='SalePrice',
).to_csv(
    'data/pca/test_y.csv',
    index=False,
    float_format="%.15f",
)

pd.Series(
    data=train_y_pred_t,
    name='SalePrice',
).to_csv(
    'data/pca/transform/train_y.csv',
    index=False,
    float_format="%.15f",
)

pd.Series(
    data=test_y_pred_t,
    name='SalePrice',
).to_csv(
    'data/pca/transform/test_y.csv',
    index=False,
    float_format="%.15f",
)

pd.Series(
    index=pd.Index(
        range(len(train_X) + 1, len(train_X) + len(test_X) + 1),
        name='Id',
    ),
    data=test_y_pred_t,
    name='SalePrice',
).to_csv(
    'output/submission5.csv',
    float_format="%.15f",
)

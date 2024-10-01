# Import:

from _imports import *

sklearn.set_config(transform_output='pandas')

# Load Data:

train_X: np.ndarray = pd.read_csv('data/train_X_prep.csv').to_numpy()
train_y: np.ndarray = pd.read_csv('data/train_y.csv').iloc[:, 0].to_numpy()
test_X: np.ndarray = pd.read_csv('data/test_X_prep.csv').to_numpy()

# Predict:

train_y_log10: np.ndarray = np.log10(train_y)
train_y_t: np.ndarray = (train_y_log10 - train_y_log10.mean()) / train_y_log10.std()

m = RandomForestRegressor(
    n_estimators=1000,
    criterion='squared_error',
    max_depth=16,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    max_samples=0.9,
    ccp_alpha=0.0,
    random_state=1,
)

m.fit(train_X, train_y_t)

ytp: np.ndarray = m.predict(test_X)

test_y_log10: np.ndarray = ytp * train_y_log10.std() + train_y_log10.mean()
test_y: np.ndarray = np.power(10.0, test_y_log10)

pd.Series(
    index=pd.Index(range(1461, 2920), name='Id'),
    data=test_y,
    name='SalePrice',
).to_csv(
    'output/submission.csv',
    index=True,
    header=True,
    float_format='%f',
)

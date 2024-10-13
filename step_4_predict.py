# Import:

from _imports import *

Path('data/predict/').mkdir(parents=True, exist_ok=True)
Path('data/predict/model/').mkdir(parents=True, exist_ok=True)
Path('data/predict/transform/').mkdir(parents=True, exist_ok=True)

keras.utils.set_random_seed(100)

# Load Data:

train_X: np.ndarray = pd.read_csv('data/preprocess/train_X.csv').to_numpy()
test_X: np.ndarray = pd.read_csv('data/preprocess/test_X.csv').to_numpy()

train_X_fixed: np.ndarray = pd.read_csv('data/impute/train_X.csv').to_numpy()
test_X_fixed: np.ndarray = pd.read_csv('data/impute/test_X.csv').to_numpy()

train_y: np.ndarray = pd.read_csv('data/preprocess/train_y.csv').iloc[:, 0].to_numpy()

# Make and Fit Model:

n_features = train_X.shape[1]

model = keras.Sequential([
    keras.layers.Input(shape=(n_features,), name='input'),
    keras.layers.Dense(
        500, activation='relu',
        kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=keras.regularizers.L2(1e-4),
        activity_regularizer=keras.regularizers.L2(1e-5),
        name='hidden1',
    ),
    keras.layers.Dropout(0.01, seed=101, name='dropout1'),
    keras.layers.Dense(
        500, activation='relu',
        kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=keras.regularizers.L2(1e-4),
        activity_regularizer=keras.regularizers.L2(1e-5),
        name='hidden3',
    ),
    keras.layers.Dropout(0.01, seed=103, name='dropout3'),
    keras.layers.Dense(1, activation='linear', name='output')
], name='model')

model.compile(
    loss=keras.losses.MeanSquaredError(name='MSE'),
    optimizer=keras.optimizers.Adam(),
    metrics=[
        keras.metrics.RootMeanSquaredError(name='RMSE'),
        keras.metrics.MeanAbsoluteError(name='MAE'),
    ]
)

model.fit(
    train_X,  # TODO: HOW ? Why not use fixed ?
    train_y,
    epochs=100,
    batch_size=8,
)

# Load Model:
# (This is commented out ...)

# model = keras.saving.load_model('data/predict/model/model.keras')

# Predict:

train_y_pred = model.predict(train_X_fixed)[:, 0]
test_y_pred = model.predict(test_X_fixed)[:, 0]

# Transform target:

target_transformer = joblib.load('data/preprocess/model/target_pipeline.pkl')

train_y_pred_t = target_transformer.inverse_transform(train_y_pred.reshape(-1, 1))[:, 0]
test_y_pred_t = target_transformer.inverse_transform(test_y_pred.reshape(-1, 1))[:, 0]

# Save Data, Metadata and Models:

model.save('data/predict/model/model.keras')

pd.Series(
    data=train_y_pred,
    name='SalePrice',
).to_csv(
    'data/predict/train_y.csv',
    index=False,
    float_format='%.15f',
)

pd.Series(
    data=test_y_pred,
    name='SalePrice',
).to_csv(
    'data/predict/test_y.csv',
    index=False,
    float_format='%.15f',
)

pd.Series(
    data=train_y_pred_t,
    name='SalePrice',
).to_csv(
    'data/predict/transform/train_y.csv',
    index=False,
    float_format='%.15f',
)

pd.Series(
    data=test_y_pred_t,
    name='SalePrice',
).to_csv(
    'data/predict/transform/test_y.csv',
    index=False,
    float_format='%.15f',
)

pd.Series(
    index=pd.Index(
        range(len(train_X) + 1, len(train_X) + len(test_X) + 1),
        name='Id',
    ),
    data=test_y_pred_t,
    name='SalePrice',
).to_csv(
    'output/submission4.csv',
    float_format='%.15f',
)

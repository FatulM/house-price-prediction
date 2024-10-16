# Import:

from _imports import *

Path('data/impute/').mkdir(parents=True, exist_ok=True)
Path('data/impute/model/').mkdir(parents=True, exist_ok=True)
Path('data/impute/pred/').mkdir(parents=True, exist_ok=True)

keras.utils.set_random_seed(1)

# Load Data:

types: pd.Series = pd.read_csv('data/fix/meta/types.csv', index_col=0).iloc[:, 0]

features: list[str] = pd.read_csv('data/preprocess/train_X.csv', nrows=10).columns.tolist()

train_X: np.ndarray = pd.read_csv('data/preprocess/train_X.csv').to_numpy()
train_y: np.ndarray = pd.read_csv('data/preprocess/train_y.csv').iloc[:, 0].to_numpy()
test_X: np.ndarray = pd.read_csv('data/preprocess/test_X.csv').to_numpy()

train_X_nas: np.ndarray = pd.read_csv('data/preprocess/na/train_X.csv').eq(1).to_numpy()
test_X_nas: np.ndarray = pd.read_csv('data/preprocess/na/test_X.csv').eq(1).to_numpy()

# Make and Fit Model:

n_features: int = train_X.shape[1]
ae_features_1: int = int(np.sum((types == 'numerical')))
ae_features_2: int = int(np.sum(types == 'ordinal'))
ae_features_3: int = n_features - ae_features_1 - ae_features_2
ae_features_4: int = 1

ae_train_X1: np.ndarray = train_X[:, :ae_features_1]
ae_train_X2: np.ndarray = train_X[:, ae_features_1:ae_features_1 + ae_features_2]
ae_train_X3: np.ndarray = train_X[:, ae_features_1 + ae_features_2:]
ae_train_X4: np.ndarray = train_y.reshape(-1, 1)

input1 = keras.layers.Input(shape=(n_features,), name='input')
hidden1 = keras.layers.Dense(
    500, activation='relu',
    kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=keras.regularizers.L2(1e-4),
    activity_regularizer=keras.regularizers.L2(1e-5),
    name='hidden1',
)(input1)
hidden1d = keras.layers.Dropout(0.01, seed=101, name='dropout1')(hidden1)
hidden5 = keras.layers.Dense(
    500, activation='relu',
    kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=keras.regularizers.L2(1e-4),
    activity_regularizer=keras.regularizers.L2(1e-5),
    name='hidden5',
)(hidden1d)
hidden5d = keras.layers.Dropout(0.01, seed=101, name='dropout5')(hidden5)
output1 = keras.layers.Dense(ae_features_1, activation='linear', name='output1')(hidden5d)
output2 = keras.layers.Dense(ae_features_2, activation='linear', name='output2')(hidden5d)
output3 = keras.layers.Dense(ae_features_3, activation='sigmoid', name='output3')(hidden5d)
output4 = keras.layers.Dense(ae_features_4, activation='linear', name='output4')(hidden5d)
model = keras.Model(inputs=input1, outputs=[output1, output2, output3, output4], name='model')

model.compile(
    loss=[
        keras.losses.MeanAbsoluteError(name='MSE'),
        keras.losses.MeanAbsoluteError(name='MSE'),
        keras.losses.BinaryCrossentropy(name='CEN'),
        keras.losses.MeanAbsoluteError(name='MSE'),
    ],
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss_weights=[0.48, 0.09, 0.39, 0.04],
)

model.fit(
    train_X,
    [ae_train_X1, ae_train_X2, ae_train_X3, ae_train_X4],
    epochs=250,
    batch_size=8,
)

# Predict for Train Data:

[ae_out1, ae_out2, ae_out3, ae_out4] = model.predict(train_X)

ae_train_X_pred: np.ndarray = np.hstack([ae_out1, ae_out2, ae_out3])
ae_train_y_pred: np.ndarray = ae_out4[:, 0]

# Predict for Test Data:

[ae_out1, ae_out2, ae_out3, ae_out4] = model.predict(test_X)

ae_test_X_pred: np.ndarray = np.hstack([ae_out1, ae_out2, ae_out3])
ae_test_y_pred: np.ndarray = ae_out4[:, 0]

# Impute Train Data:

train_X_impute: np.ndarray = train_X.copy()
train_X_impute[train_X_nas] = ae_train_X_pred[train_X_nas]

# Impute test Data:

test_X_impute: np.ndarray = test_X.copy()
test_X_impute[test_X_nas] = ae_test_X_pred[test_X_nas]

# Save Data, Metadata and Models:

model.save('data/impute/model/model.keras')

pd.DataFrame(
    columns=features,
    data=train_X_impute,
).to_csv(
    'data/impute/train_X.csv',
    index=False,
    float_format='%.15f',
)

pd.DataFrame(
    columns=features,
    data=test_X_impute,
).to_csv(
    'data/impute/test_X.csv',
    index=False,
    float_format='%.15f',
)

pd.DataFrame(
    columns=features,
    data=ae_train_X_pred,
).to_csv(
    'data/impute/pred/train_X.csv',
    index=False,
    float_format='%.15f',
)

pd.Series(
    data=ae_train_y_pred,
    name='SalePrice',
).to_csv(
    'data/impute/pred/train_y.csv',
    index=False,
    float_format='%.15f',
)

pd.DataFrame(
    columns=features,
    data=ae_test_X_pred,
).to_csv(
    'data/impute/pred/test_X.csv',
    index=False,
    float_format='%.15f',
)

pd.Series(
    data=ae_test_y_pred,
    name='SalePrice',
).to_csv(
    'data/impute/pred/test_y.csv',
    index=False,
    float_format='%.15f',
)

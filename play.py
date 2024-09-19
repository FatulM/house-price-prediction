# Import:

from _imports import *

# Load Data:

train_X: pd.DataFrame = pd.read_csv('data/train_X_prep.csv').to_numpy()
train_y: np.ndarray = pd.read_csv('data/train_y.csv').iloc[:, 0].to_numpy()
test_X: pd.DataFrame = pd.read_csv('data/test_X_prep.csv').to_numpy()

# TODO:

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)

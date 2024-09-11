# Import:

from _imports import *

# Load Meta Data:

types: dict[str, str] = pd.read_csv('data/meta/types.csv', index_col=0).iloc[:, 0].to_dict()
discrete: dict[str, list[str]] = pd.read_csv('data/meta/discrete.csv', index_col=0).iloc[:, 0] \
    .map(lambda x: x.split(sep="|")).to_dict()

# Load Data:

train_X: pd.DataFrame = pd.read_csv('data/train_X.csv')
train_y: np.ndarray = pd.read_csv('data/train_y.csv').iloc[:, 0].to_numpy()
test_X: pd.DataFrame = pd.read_csv('data/test_X.csv')

# TODO:

print()

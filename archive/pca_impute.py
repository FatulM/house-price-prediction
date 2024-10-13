import math
from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA


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

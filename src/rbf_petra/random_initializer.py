import numpy as np
import pandas as pd
from keras.initializers import Initializer


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])

        # type checking to access elements of data correctly
        if type(self.X) == np.ndarray:
                return self.X[idx, :]
        elif type(self.X) == pd.core.frame.DataFrame:
                return self.X.iloc[idx, :]
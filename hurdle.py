from typing import Optional, Union
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from lightgbm import LGBMClassifier, LGBMRegressor


class HurdleRegression(BaseEstimator):
    """ Regression model which handles excessive zeros by fitting a two-part model and combining predictions:
            1) binary classifier
            2) continuous regression
    Implementeted as a valid sklearn estimator, so it can be used in pipelines and GridSearch objects.
    Args:
        clf_name: currently supports either 'logistic'
        reg_name: currently supports either 'linear'
        clf_params: dict of parameters to pass to classifier sub-model when initialized
        reg_params: dict of parameters to pass to regression sub-model when initialized
    """

    def __init__(self,
                 clf_name: str = 'logistic',
                 reg_name: str = 'linear',
                 clf_params: Optional[dict] = None,
                 reg_params: Optional[dict] = None):

        self.clf_name = clf_name
        self.reg_name = reg_name
        self.clf_params = clf_params
        self.reg_params = reg_params

    @staticmethod
    def _resolve_estimator(func_name: str):
        """ Lookup table for supported estimators.
        This is necessary because sklearn estimator default arguments
        must pass equality test, and instantiated sub-estimators are not equal. """

        funcs = {'linear': LinearRegression(),
                 'logistic': LogisticRegression(solver='liblinear'),
#                  'LGBMRegressor': LGBMRegressor(n_estimators=50),
#                  'LGBMClassifier': LGBMClassifier(n_estimators=50)
                }

        return funcs[func_name]

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]):
        X, y = check_X_y(X, y, dtype=None,
                         accept_sparse=False,
                         accept_large_sparse=False,
                         force_all_finite='allow-nan')

        if X.shape[1] < 2:
            raise ValueError('Cannot fit model when n_features = 1')

        self.clf_ = self._resolve_estimator(self.clf_name)
        if self.clf_params:
            self.clf_.set_params(**self.clf_params)
        self.clf_.fit(X, y > 0)

        self.reg_ = self._resolve_estimator(self.reg_name)
        if self.reg_params:
            self.reg_.set_params(**self.reg_params)
        self.reg_.fit(X[y > 0], y[y > 0])

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        """ Predict combined response using binary classification outcome """
        X = check_array(X, accept_sparse=False, accept_large_sparse=False)
        check_is_fitted(self, 'is_fitted_')
        return self.clf_.predict(X) * self.reg_.predict(X)

    def predict_expected_value(self, X: Union[np.ndarray, pd.DataFrame]):
        """ Predict combined response using probabilistic classification outcome """
        X = check_array(X, accept_sparse=False, accept_large_sparse=False)
        check_is_fitted(self, 'is_fitted_')
        return self.clf_.predict_proba(X)[:, 1] * self.reg_.predict(X)


def manual_test():
    """ Validate estimator using sklearn's provided utility and ensure it can fit and predict on fake dataset. """
    check_estimator(HurdleRegression)
    from sklearn.datasets import make_regression
    X, y = make_regression()
    reg = HurdleRegression()
    reg.fit(X, y)
    reg.predict(X)


if __name__ == '__main__':
    manual_test()
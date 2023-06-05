import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin

class PLS_DA:
    """Partial Least Squares Discriminant Analysis.

    Partial Least Squares Discriminant Analysis (PLS-DA) is a multivariate statistical
    technique that extends Partial Least Squares (PLS) regression to address 
    classification problems. It combines the principles of PLS regression and
    traditional linear discriminant analysis to find a linear combination of predictor
    variables that maximizes the separation between classes. PLS-DA is particularly 
    useful when dealing with high-dimensional datasets and when there are correlations 
    between the predictor variables.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.
        This should be left to None if `covariance_estimator` is used.
        Note that shrinkage works only with 'lsqr' and 'eigen' solvers.
    """

    def __init__(self, n_components, shrinkage):
        # Dimensionality Reduction
        self.pls = PLSRegression(n_components=n_components)

        # Classification
        self.clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)

    def fit(self, X_train, y_train):
        """Fit model to data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
                  Training vectors.

        y_train : array-like of shape (n_samples,)
                  Target vectors.
        
        Returns
        -------
        self : object
            Fitted model.
        """

        X_train, _ = self.pls.fit_transform(X_train, y_train)

        self.clf.fit(X_train, y_train)

        return self

    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """

        X = self.pls.transform(X)

        y_pred = self.clf.predict(X)

        return y_pred
    
    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated probabilities.
        """

        X = self.pls.transform(X)

        C = self.clf.predict_proba(X)

        return C
    
    def get_params(self, deep):
        return {'n_components': self.pls.get_params(deep=deep)['n_components'],
                'shrinkage': self.clf.get_params(deep=deep)['shrinkage']}

def correlation(X, y):
    """Calculates Pearson correlation coefficients.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Input data.

    y: array-like of shape (n_samples,)
       Target vectors.
    
    Returns
    -------
    corr_coefs : array-like of shape (n_features,)
        Pearson correlation coefficient of every feature with the target vector.
    """

    X = np.asarray(X)
    
    y = np.asarray(y)
    y_mean, y_std = np.mean(y), np.std(y)

    # Initialize empty array of shape (n_features,)
    corr_coefs = np.zeros((X.shape[1],))

    # Calculate Pearson correlation coefficient for every feature
    for i in range(X.shape[1]):
        x = X[:, i]
        x_mean, x_std = np.mean(x), np.std(x)

        corr_xy = np.sum((x-x_mean)*(y-y_mean))/(x_std*y_std)
        corr_xy /= x.size

        corr_coefs[i] = corr_xy

    return corr_coefs

class SPCA_LR:
    """Supervised Principal Component Analysis + Linear Regression.

    Supervised Principal Components is a statistical technique use for regression 
    analysis. The approach is similar to Principal Component Regression (PCR), 
    with the key distinction being that it only considers the relevant 
    predictor variables before conducting Principal Component Analysis (PCA). 
    This ensures that the regression models are built using the most relevant 
    and informative features.

    Parameters
    ----------
    top_n : int or "all"
        Number of top features to select. 
        The "all" option bypasses selection, for use in a parameter search.

    pca_components : 'int or 'mle'
        Number of components to keep. Should be in `[1, min(n_samples, top_n)]`.
        If ``pca_components == 'mle'`` Minka's MLE is used to guess the dimension.
    """

    def __init__(self, top_n, pca_components):
        # Feature Selection
        self.fs = SelectKBest(score_func=correlation, k=top_n)

        # Dimensionality Reduction
        self.pca = PCA(n_components=pca_components)

        # Regression
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        """Fit model to data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
                  Training vectors.

        y_train : array-like of shape (n_samples,)
                  Target vectors.
        
        Returns
        -------
        self : object
            Fitted model.
        """

        X_train = self.fs.fit_transform(X_train, y_train)

        X_train = self.pca.fit_transform(X_train)

        self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """

        X = self.fs.transform(X)

        X = self.pca.transform(X)

        y_pred = self.model.predict(X)

        return y_pred
    
    def score(self, X, y):
        """Return Negative Mean Square Error as score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target vectors.

        Returns
        -------
        neg_mse: float
                 Negative Mean Square Error of the model's predictions.
        """

        y_pred = self.predict(X)

        neg_mse = -mean_squared_error(y, y_pred)

        return neg_mse
    
    def get_params(self, deep):
        return {'top_n': self.fs.get_params(deep=deep)['k'],
                'pca_components': self.pca.get_params(deep=deep)['n_components']}
    
class PCR:
    """Principal Component Regression.

    Principal Component Regression (PCR) is a statistical technique that combines
    Principal Component Analysis (PCA) with linear regression. It is used to handle 
    multicollinearity (high correlation) among predictor variables in regression 
    analysis.

    Parameters
    ----------
    n_components : 'int or 'mle'
        Number of components to keep. Should be in `[1, min(n_samples, n_features)]`.
        If ``n_components == 'mle'`` Minka's MLE is used to guess the dimension.
    """

    def __init__(self, n_components):
        # Dimensionality Reduction
        self.pca = PCA(n_components=n_components)

        # Regression
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        """Fit model to data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
                  Training vectors.

        y_train : array-like of shape (n_samples,)
                  Target vectors.
        
        Returns
        -------
        self : object
            Fitted model.
        """

        X_train = self.pca.fit_transform(X_train)

        self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """

        X = self.pca.transform(X)

        y_pred = self.model.predict(X)

        return y_pred
    
    def score(self, X, y):
        """Return Negative Mean Square Error as score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target vectors.

        Returns
        -------
        neg_mse: float
                 Negative Mean Square Error of the model's predictions.
        """

        y_pred = self.predict(X)

        neg_mse = -mean_squared_error(y, y_pred)

        return neg_mse
    
    def get_params(self, deep):
        return {'n_components': self.pca.get_params(deep=deep)['n_components']}
    
    
class SFS_LR:
    """Sequential Feature Selection - Linear Regression.

    Sequential Feature Selection (SFS) is a feature selection technique used
    to iteratively build a subset of features by sequentially adding or removing 
    features based on a specific criterion. Forward selection is one variant of 
    SFS, where features are added one at a time in a forward manner until a 
    stopping criteria is met. The selected features are then used to build a linear
    regression model with L2 penalty.

    Parameters
    ----------
    n_features_to_select : "auto", int or float, default='warn'
        If `"auto"`, the behaviour depends on the `tol` parameter:

        - if `tol` is not `None`, then features are selected until the score
          improvement does not exceed `tol`.
        - otherwise, half of the features are selected.

        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

    alpha : {float, ndarray of shape (n_targets,)}, default=0.0
        Constant that multiplies the L2 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

        When `alpha = 0`, the objective is equivalent to ordinary least
        squares, solved by the :class:`LinearRegression` object. For numerical
        reasons, using `alpha = 0` with the `Ridge` object is not advised.
        Instead, you should use the :class:`LinearRegression` object.

        If an array is passed, penalties are assumed to be specific to the
        targets. Hence they must correspond in number.
    """

    def __init__(self, n_features_to_select=8, alpha=0):
        # Feature Selection
        self.fs = SequentialFeatureSelector(estimator=LinearRegression(),
                                            n_features_to_select=n_features_to_select,
                                            direction='forward',
                                            scoring='r2',
                                            cv=3,
                                            n_jobs=-1)
        
        # Regression
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train, y_train):
        """Fit model to data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
                  Training vectors.

        y_train : array-like of shape (n_samples,)
                  Target vectors.
        
        Returns
        -------
        self : object
            Fitted model.
        """

        X_train = self.fs.fit_transform(X_train, y_train)

        self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """

        X = self.fs.transform(X)

        y_pred = self.model.predict(X)

        return y_pred
    
    def score(self, X, y):
        """Return Negative Mean Square Error as score..

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target vectors.

        Returns
        -------
        neg_mse: float
                 Negative Mean Square Error of the model's predictions.
        """

        y_pred = self.predict(X)

        neg_mse = -mean_squared_error(y, y_pred)

        return neg_mse

    def get_params(self, deep):
        return {'n_features_to_select': self.fs.get_params(deep=deep)['n_features_to_select'],
                'alpha': self.model.get_params(deep=deep)['alpha']}
    
def quadratic_transform(X):
    """Quadratic Transformation

    A quadratic transform is a technique used to introduce (non-linearity) 
    quadratic terms or interactions between features in a dataset. It involves 
    creating new features by taking the square or product of existing features, 
    allowing for more complex relationships to be captured by a model. 
    Quadratic transform increases the dimensionality of the feature space, 
    which may require careful consideration of computational resources and 
    potential overfitting.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Input data.

    Returns
    -------
    X_2: array-like of shape (n_samples, n_features*(n_features-1)//2 + 2*n_features)
         Quadratically transformed input data.
    """

    X_2 = X_2 = np.zeros((X.shape[0], X.shape[1]*(X.shape[1]-1)//2 + 2*X.shape[1] + 1))

    for i in range(X.shape[0]):
        x = [1] + list(X[i, :])
        x_2 = []
        for j in range(len(x)):
            deg_1 = x[j]
            for k in range(j, len(x)):
                deg_2 = deg_1*x[k]
                x_2.append(deg_2)

        X_2[i, :] = x_2

    return X_2[:, 1:]
    
class SFS_QR(BaseEstimator, RegressorMixin):
    """Sequential Feature Selection - Quadratic Regression.

    Quadratic transformation is carried out before performing feature selection using SFS.
    Sequential Feature Selection (SFS) is a feature selection technique used
    to iteratively build a subset of features by sequentially adding or removing 
    features based on a specific criterion. Forward selection is one variant of 
    SFS, where features are added one at a time in a forward manner until a 
    stopping criteria is met. The selected features are then used to build a linear
    regression model with L2 penalty.

    Parameters
    ----------
    n_features_to_select : "auto", int or float, default='warn'
        If `"auto"`, the behaviour depends on the `tol` parameter:

        - if `tol` is not `None`, then features are selected until the score
          improvement does not exceed `tol`.
        - otherwise, half of the features are selected.

        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

    alpha : {float, ndarray of shape (n_targets,)}, default=0.0
        Constant that multiplies the L2 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

        When `alpha = 0`, the objective is equivalent to ordinary least
        squares, solved by the :class:`LinearRegression` object. For numerical
        reasons, using `alpha = 0` with the `Ridge` object is not advised.
        Instead, you should use the :class:`LinearRegression` object.

        If an array is passed, penalties are assumed to be specific to the
        targets. Hence they must correspond in number.
    """

    def __init__(self, n_features_to_select=8, alpha=0):
        # Transformation
        self.transform = quadratic_transform

        # Feature Selection
        self.fs = SequentialFeatureSelector(estimator=LinearRegression(),
                                            n_features_to_select=n_features_to_select,
                                            direction='forward',
                                            scoring='r2',
                                            cv=3,
                                            n_jobs=-1)
        
        # Regression
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train, y_train):
        """Fit model to data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
                  Training vectors.

        y_train : array-like of shape (n_samples,)
                  Target vectors.
        
        Returns
        -------
        self : object
            Fitted model.
        """

        X_train = self.transform(X_train)

        X_train = self.fs.fit_transform(X_train, y_train)

        self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """

        X = self.transform(X)

        X = self.fs.transform(X)

        y_pred = self.model.predict(X)

        return y_pred
    
    def score(self, X, y):
        """Return Negative Mean Square Error as score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target vectors.

        Returns
        -------
        neg_mse: float
                 Negative Mean Square Error of the model's predictions.
        """

        y_pred = self.predict(X)

        neg_mse = -mean_squared_error(y, y_pred)

        return neg_mse

    def get_params(self, deep):
        return {'n_features_to_select': self.fs.get_params(deep=deep)['n_features_to_select'],
                'alpha': self.model.get_params(deep=deep)['alpha']}
    

class PLSR(BaseEstimator, RegressorMixin):
    """Partial Least Squares Regression.

    Partial Least Squares Regression (PLSR) is a statistical technique used 
    for regression analysis, especially when dealing with high-dimensional 
    predictor variables and multicollinearity. PLSR combines the features of 
    Partial Least Squares (PLS) and regression to model the relationship between 
    the predictor and the target variable.

    Parameters
    ----------
    n_components : int
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.
    """

    def __init__(self, n_components):
        # Regression
        self.model = PLSRegression(n_components=n_components)

    def fit(self, X_train, y_train):
        """Fit model to data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
                  Training vectors.

        y_train : array-like of shape (n_samples,)
                  Target vectors.
        
        Returns
        -------
        self : object
            Fitted model.
        """

        self.model.fit(X_train, y_train)
        
        return self

    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        
        y_pred = self.model.predict(X).squeeze()

        return y_pred

    def get_params(self, deep):
        return {'n_components': self.model.get_params(deep=deep)['n_components']}
    




__author__ = 'yuxinsun'
import numpy as np
from cvxopt import *
from sklearn.base import BaseEstimator


def LPcvx(z, y, D):
    """ Linear programming optimisation for LPBoost

    Parameters
    -------
    :param z: array_like, shape (n_iterations, n_samples)
        transposed hypothesis space in current iteration
    :param y: array_like, shape (n_samples, )
        desired labels for classification
    :param D: float
        optimisation parameter, practically D = 1/(n_samples, nu)
    Return
    -------
    :return u: array_like, shape (n_samples, )
        misclassification cost
    :return beta: float
        beta in LPBoost
    :return c4.multiplier.value: array_like, shape (n_features, )
        weights of weak learners
    """

    m = y.size

    # Initialise
    u = modeling.variable(m, 'u')
    u.value = matrix(np.ones(m)/m)
    beta = modeling.variable(1, 'beta')
    beta.value = 0

    # Constraints
    c1 = (modeling.sum(u) == 1)
    c2 = (u <= D)
    c3 = (u >= 0)
    c4 = (modeling.dot(matrix(z), u) <= beta)

    # Solve optimisation problems
    lp = modeling.op(beta, [c1, c2, c3, c4])
    solvers.options['show_progress']=False
    sol = lp.solve()

    return u.value, beta.value, c4.multiplier.value


class lpboost(BaseEstimator):
    """
    Linear programming boosting (LPBoost) model
    Representation of a LPBoost model.
    This class allows for selecting weak learners (features) from an explicit hypothesis space.
    Parameters
    -------
    nu: float between 0 and 1, exclusive. optional
        Optimisation parameter in LPBoost, usually used to control D
    threshold: float, optional
        Threshold of feature selection. Features with weights below threshold would be discarded.
    n_iter: int, optional
        Maximum iteration of LPBoost
    verbose: int, default 0
        Enable verbose output. If greater than 0 then it prints the iterations in fit() and fit_transform().
    Attributes
    -------
    converged: bool
        True when convergence reached in fit() and fit_transform().
    u: array_like, shape (n_samples, )
        Misclassification cost
    a: array_like, shape (n_selected_features, )
        Weights of selected features, such features are selected because corresponding weights are lower than threshold.
    beta: float
        beta in LPBoost
    idx: list of integers
        Indices of selected features
    """
    def __init__(self, nu=0.5, threshold=10**-3, n_iter=1000, verbose=0):
        self.nu = nu
        self.threshold = threshold
        self.n_iter = n_iter
        self.verbose = verbose
        self.converged = False

        if n_iter < 1:
            raise ValueError('LPBoost requires at lease one iteration.')

        if nu <= 0 or nu > 1:
            raise ValueError('Invalid value for nu = %10.2f.' % nu)

    def _transform(self, X):
        """
        Transform data matrix to a matrix with selected features only
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_features)
            Data matrix of the explicit hypothesis space. each column corresponds to a weak learner/ feature
        Returns
        -------
        :return: array_like, shape (n_samples, n_selected_features)
            Data matrix whose columns are selected weak learners by LPBoost
        """
        return X[:, self.idx]

    def transform(self, X):
        """
        Transform data matrix to a matrix with selected features only. Calls _transform() to transform data
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_features)
            Data matrix of the explicit hypothesis space. each column corresponds to a weak learner/ feature
        Returns
        -------
        :return: array_like, shape (n_samples, n_selected_features)
            Data matrix whose columns are selected weak learners by LPBoost
        """
        return self._transform(X)

    def predict(self, X):
        """
        Predict labels given a data matrix by LPBoost classifier: sign(data_transformed * a)
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_selected_features)
            Data matrix to be predicted
        Returns
        -------
        :return: array_like, shape (n_samples, )
            Predicted labels
        """
        X_tran = self.transform(X)
        return np.sign(np.dot(X_tran, self.a))

    def _fitString(self, X, y):
        """
        Perform LPBoost on string features. Usually a l2 normalisation is performed. If the hypothesis space contains
        positive/ negative features only, then the space needs to be duplicated by its additive inverse. This is to
        ensure the performance of LPBoost.
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_features)
            Data matrix of explicit features
        :param y: array_like, shape (n_samples,)
            Desired labels
        Returns
        -------
        :return:
        """
        n_samples = y.size
        u = np.ones(n_samples)/n_samples
        D = 1./(float(n_samples)*self.nu)
        beta = 0.
        counter = 1

        hypo = np.dot(np.multiply(u, y), X)
        idx = [np.argmax(hypo)]
        crit = np.max(hypo)
        F = np.asarray([X[:, idx[-1]]*y])

        while crit >= beta + 10**-6 and counter <= self.n_iter:
            [u, beta, a] = LPcvx(F.transpose(), y, D)
            hypo = np.dot(np.multiply(np.squeeze(u), y), X)
            idx.append(np.argmax(hypo))
            crit = np.max(hypo)
            beta = np.asarray(beta)[0]
            F = np.append(F, [X[:, idx[-1]]*y], axis=0)

            if self.verbose > 0:
                print('Iteration: %d, beta %10.6f, criterion: %10.6f.' % (counter, beta, crit))

            counter += 1

        if counter - 1 < self.n_iter:
            self.converged = True

        a = np.asarray(a)
        self.u = u
        self.a = a[np.where(a >= self.threshold)]
        self.beta = beta
        self.idx = [idx[i] for i in np.where(a >= self.threshold)[0].tolist()]

    def fit(self, X, y):
        """
        Fit LPBoost model to data.
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_features)
            Data matrix of explicit features
        :param y: array_like, shape (n_samples,)
            Desired labels
        Returns:
        -------
        :return:
        """
        return self._fitString(X, y)

    def fit_transform(self, X, y):
        """
        Fit data to LPBoost model and transform data.
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_features)
            Data matrix with explicit features
        :param y: array_like, shape (n_samples,)
            Desired labels
        Returns
        -------
        :return: array_like, shape (n_samples, n_selected_features)
            Transformed data matrix with columns being the selected features
        """
        self._fitString(X, y)
        return self._transform(X)

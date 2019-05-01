import numpy as np
import pandas as pd
import scipy

class PCR(BasicRegression):

    def __init__(self):
        pass

    def fit(self, X, Y, max_iter = 1000, eps = 1e-6):
        """ Fitting the data with Principal Components Regression
            N:          the number of data
            Mx:         the number of variable in X
            My:         the number of variable in Y
            X :         N x Mx nparray
            Y :         N * My nparray
            max_iter:   the maximum number of iteratons
            eps:        the maximum tolerant of diff. between iterations
        """
        pass

    def predict(self, X):
        """ Predicting Y for X with Principal Components Regression
            N_pred:     the number of data for prediction
            X :         N_pred x Mx nparray or Mx nparray
            Y :         N_pred * My nparray or My nparray
        """
        pass
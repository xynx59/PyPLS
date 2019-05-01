import numpy as np

class BasicRegression(object):

    def __init__(self):
        pass

    def fit(self, X, Y, max_iter = 1000, eps = 1e-6):
        """ Fitting the data
            N:          the number of data
            Mx:         the number of      in X
            My:         the number of      in Y
            X :         N x Mx nparray
            Y :         N * My nparray
            max_iter:   the maximum number of iteratons
            eps:        the maximum tolerant of diff. between iterations
        """
        pass


    def preprocessing(self, X, Y):
        """ Normalizing the data and save key information for fitting
            X_norm:         normalized X
            self.X_scale:   scale for X normalization
            self.X_center:  center for X normalization
            Y_norm:         normalized Y
            self.Y_scale:   scale for Y normalization
            self.Y_center:  center for Y normalization
            
            return X_norm, Y_norm
        """
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        if len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1))
        self.N = X.shape[0]
        self.Mx = X.shape[1]
        self.My = Y.shape[1]
        if self.N != Y.shape[0]:
            raise Exception("Data sizes of X and Y do not match.")
        
        self.X_center = X.mean(0)
        self.X_std = X.std(0, ddof = 1)
        self.X_scale = 1.0 / self.X_std
        X_norm = (X - self.X_center) * self.X_scale
        
        self.Y_center = Y.mean(0)
        self.Y_std = Y.std(0, ddof = 1)
        self.Y_scale = 1.0 / self.Y_std
        Y_norm = (Y - self.Y_center) * self.Y_scale
        
        return X_norm, Y_norm

    def predict(self, X):
        """ Predicting Y for X
            N_pred:     the number of data for prediction
            X :         N_pred x Mx nparray or Mx nparray
            Y :         N_pred * My nparray or My nparray
        """
        pass
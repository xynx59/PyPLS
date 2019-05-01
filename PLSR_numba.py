import numpy as np
import copy
from sklearn.preprocessing import normalize
import BasicRegression as BR
import numexpr as ne
import numba
from numba import jit

@jit(cache=True)
def fit_numba(X_norm, Y_norm, d, max_iter, eps, N, Mx, My):
    """ Use Numba to speed up the code. However, it seems numba is far slower than original numpy calculation.
    """
    P = np.empty((Mx, d))
    W = np.empty((Mx, d))
    Q = np.empty((My, d))
    U = np.empty((N, d))
    T = np.empty((N, d))
    b = np.empty((d))
    
    for it in range(d):
        u_tmp = Y_norm[:,np.random.randint(My)].reshape(N,1)
#            bool_eps = False
        
        for jt in range(max_iter):
            p_tmp = np.matmul(X_norm.T, u_tmp)
            p_tmp /= np.linalg.norm(p_tmp, 2)
            t_tmp = np.matmul(X_norm, p_tmp)
            q_tmp = np.matmul(Y_norm.T, t_tmp)
            q_tmp /= np.linalg.norm(q_tmp, 2)
            u_new = np.matmul(Y_norm, q_tmp)
            if np.linalg.norm(u_tmp-u_new, 2) <= eps:
                u_tmp = u_new
#                    bool_eps = True
                break
            u_tmp = u_new
        
        W[:,it] = p_tmp.reshape((Mx))
        Q[:,it] = q_tmp.reshape((My))
        U[:,it] = u_tmp.reshape((N))
        T[:,it] = t_tmp.reshape((N))
        t_sum = np.sum(t_tmp*t_tmp)
        b[it] = np.sum(u_tmp*t_tmp) / t_sum
        P[:,it] = (np.matmul(np.transpose(X_norm), t_tmp) / t_sum).reshape((Mx))
        X_norm -= np.matmul(t_tmp, P[:,it].reshape(1,Mx))

        Y_norm -= b[it] * np.matmul(t_tmp, q_tmp.T)      
#            if bool_eps == False:
#                self.d = it+1
#                break
    return W, P, Q, U, T, b

class PLSR(BR.BasicRegression):

    def __init__(self, n_components = None):
        self.n_components = n_components
        pass

    def fit(self, X, Y, d = None, max_iter = 500, eps = 1e-6):
        """ Fitting the data with Principal Least Squares Regression
            N:          the number of data
            Mx:         the number of variable in X
            My:         the number of variable in Y
            X :         N x Mx nparray
            Y :         N * My nparray
            max_iter:   the maximum number of iteratons
            eps:        the maximum tolerant of diff. between iterations
        """
        
        X_norm, Y_norm = self.preprocessing(X, Y)
        
        if d == None:
            d = self.n_components
        if d == None:
            d = np.min((self.Mx, self.N))
        else:
            d = np.min([self.N, d, self.Mx])
        self.d = d
        
        W, P, Q, U, T, b = fit_numba(X_norm, Y_norm, d = self.d, max_iter = max_iter, eps = eps, N = self.N, Mx = self.Mx, My = self.My)
            
        self.P = P[:,0:self.d]
        self.W = W[:,0:self.d]
        self.Q = Q[:,0:self.d]
        self.U = U[:,0:self.d]
        self.T = T[:,0:self.d]
        self.b = b[0:self.d]
        self.B = np.matmul( \
                    np.matmul(self.W, np.linalg.inv(np.matmul(self.P.T, self.W))), \
                    np.matmul(np.diag(self.b), self.Q.T) \
                    )

    def predict(self, X):
        """ Predicting Y for X with Principal Least Squares Regression
            N_pred:     the number of data for prediction
            X :         N_pred x Mx nparray or Mx nparray
            Y :         N_pred * My nparray or My nparray
        """
        
        typeX = type(X)
        if typeX == list:
            X = np.array(X)
        elif typeX == np.float or typeX == float or typeX == np.int or typeX == int:
            X = np.array([X])
        if len(X.shape) == 1:
            Y = self.Y_center + \
                np.matmul(((X - self.X_center)*self.X_scale).T, self.B) \
                * self.Y_std
        else:
            N_pred = X.shape[0]
            Y = np.empty((N_pred, self.My))
            for it in range(N_pred):
                Y[it,:] = self.Y_center + \
                    np.matmul(((X[it,:] - self.X_center)*self.X_scale).T, self.B) \
                    * self.Y_std
        
        return Y
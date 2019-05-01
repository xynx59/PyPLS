# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:14:16 2019

@author: PRJ
"""

import os
import numpy as np
import pandas as pd
import PLSR
import FillMissing as FM
from sklearn.model_selection import KFold



def CV_Regression(X, Y, K = 10, missing = None, Model = PLSR.PLSR, \
                  d_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, \
                            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]):
    """
        Cross-Validation for selecting the best 
    """
    [N, Mx] = X.shape
    if len(Y.shape) == 1:
        Y = Y.reshape(N,-1)
    My = Y.shape[1]
    
    X_center = X.mean(0)
    X_std = X.std(0, ddof = 1)
    X_scale = 1.0 / X_std
    Y_center = Y.mean(0)
    Y_std = Y.std(0, ddof = 1)
    Y_scale = 1.0 / Y_std
    
    if not (missing == None):
        X = FM.fillmissing(X, missing)
    d_max = np.min([Mx, N])
    if (type(d_list[0]) == np.float or type(d_list[0]) == float):
        d_list = np.array(d_list)
        d_list = np.unique(np.array(d_list*d_max, dtype = np.int))
        if d_list[0] == 0:
            d_list = d_list[1:]
    Md = d_list.shape[0]
    Err_list = np.zeros(Md, dtype = np.float)
    
    kf = KFold(n_splits = K)
    kf.get_n_splits(X)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        if not (missing == None):
            Y_train = FM.fillmissing(Y_train, missing)
        
        for dt in range(Md):
            d = d_list[dt]
            model = Model(n_components = d)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            N_test = Y_pred.shape[0]
            for it in range(N_test):
                for jt in range(My):
                    if np.isnan(Y_test[it,jt]):
                        continue
                    Err_list[dt] += ((Y_pred[it,jt]-Y_test[it,jt])*Y_scale[jt]) ** 2
    idx_min = np.argmin(Err_list)
    return d_list[idx_min]
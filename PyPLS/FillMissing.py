# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:38:13 2019

@author: PRJ
"""

import numpy as np
import copy

def fillmissing(X_raw, missing = 'median'):
    """ 
        Fill missing value with specific value
    """
    X = copy.deepcopy(X_raw)
    if len(X.shape) == 1:
        N = X.shape[0]
        if missing == 'mean':
            X_fill = np.nanmean(X)
        elif missing == 'median':
            X_fill = np.nanmedian(X)
        elif missing == 'max':
            X_fill = np.nanmax(X)
        elif missing == 'min':
            X_fill = np.nanmin(X)
        else:
            X_fill = 0
        for it in range(N):
            X[it] = X_fill
    else:
        [N, M] = X.shape
        if missing == 'mean':
            X_fill = np.nanmean(X, axis = 0)
        elif missing == 'median':
            X_fill = np.nanmedian(X, axis = 0)
        elif missing == 'max':
            X_fill = np.nanmax(X, axis = 0)
        elif missing == 'min':
            X_fill = np.nanmin(X, axis = 0)
        else:
            X_fill = np.zeros(M)
        for it in range(N):
            for jt in range(M):
                if np.isnan(X[it,jt]):
                    X[it,jt] = X_fill[jt]

    return X
    
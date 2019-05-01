# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:51:57 2019

@author: PRJ
"""

import os
import numpy as np
import pandas as pd
from PyPLS import PLSR, PLSR_numba

import CV
import time
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def data_generation(dataset = 'house', datatype = 'csv', K = 10):
    cwd =  os.getcwd()
    X = np.loadtxt(open(cwd+'/data/'+dataset+'_X.'+datatype, "rb"), delimiter=",")
    Y = np.loadtxt(open(cwd+'/data/'+dataset+'_Y.'+datatype, "rb"), delimiter=",")
    if not os.path.exists(cwd+'/data/'+dataset+'/'):
        os.mkdir(cwd+'/data/'+dataset+'/')
    kf = KFold(n_splits = K)
    kf.get_n_splits(X)
    it = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        np.savetxt(cwd+'/data/'+dataset+'/train'+'_'+str(it)+'_X.'+datatype, X_train, delimiter=",")
        np.savetxt(cwd+'/data/'+dataset+'/train'+'_'+str(it)+'_Y.'+datatype, Y_train, delimiter=",")
        np.savetxt(cwd+'/data/'+dataset+'/test'+'_'+str(it)+'_X.'+datatype, X_test, delimiter=",")
        np.savetxt(cwd+'/data/'+dataset+'/test'+'_'+str(it)+'_Y.'+datatype, Y_test, delimiter=",")
        
        it += 1


        
def main():
    cwd =  os.getcwd()
    model_list = [PLSRegression, PLSR.PLSR, LinearRegression, 'PLSR_fix']
    N_model = len(model_list)
    dataset = 'sim'
    datatype = 'csv'
    N_dataset = 5
    K = 5
    d_mat = np.zeros((N_dataset, N_model), dtype = np.int)
    err_mat = np.zeros((N_dataset, N_model), dtype = np.float)
    Err_mix = np.zeros(N_model, dtype = np.float) #Frobenius norm
    Time_mix = np.zeros(N_model, dtype = np.float) #Frobenius norm
    
    
    data_generation(dataset, datatype, N_dataset)
    
    for it in range(N_dataset):
        print (it)
        X = np.loadtxt(open(cwd+'/data/'+dataset+'/train'+'_'+str(it)+'_X.'+datatype, "rb"), delimiter=",")
        Y = np.loadtxt(open(cwd+'/data/'+dataset+'/train'+'_'+str(it)+'_Y.'+datatype, "rb"), delimiter=",")
        X_test = np.loadtxt(open(cwd+'/data/'+dataset+'/test'+'_'+str(it)+'_X.'+datatype, "rb"), delimiter=",")
        Y_test = np.loadtxt(open(cwd+'/data/'+dataset+'/test'+'_'+str(it)+'_Y.'+datatype, "rb"), delimiter=",")
        
        Mx = X.shape[1]
        Y_center = Y.mean(0)
        Y_std = Y.std(0, ddof = 1)
        Y_scale = 1.0 / Y_std
        
        for jt in range(N_model):
            T_start = time.time()
            model = model_list[jt]
            if model == LinearRegression:
                lr = model()
                lr.fit(X,Y)
                Y_pred = lr.predict(X_test)
            elif type(model) == str:
                plsr = PLSRegression(n_components = int(0.3*Mx))
                plsr.fit(X,Y)
                Y_pred = plsr.predict(X_test)
            else:
                d_cv = CV.CV_Regression(X, Y, K, None, model)
                plsr = model(n_components = d_cv)
                plsr.fit(X,Y)
                Y_pred = plsr.predict(X_test)
            T_end = time.time()
            err_mat[it, jt] = np.sum(((Y_pred-Y_test)*Y_scale)**2)
            d_mat[it, jt] = d_cv
            Time_mix[jt] += T_end-T_start
    Err_mix = np.sqrt(np.sum(err_mat, axis = 0))
    print (Err_mix)
    print (Time_mix)
    print (d_mat)
    
if __name__ == "__main__":
    main()
    
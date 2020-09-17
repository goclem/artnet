#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Convolutional network example on the MINST dataset
Author: Clement Gorin
Contact: gorin@gate.cnrs.fr
Date: September 2020
"""

#%% Modules

import os
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Flatten

#%% Data

# Loading
X, y = load_boston(return_X_y=True)

# Normalising    
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1,1)).reshape(-1)

# K fold
kf_cv = KFold(n_splits=10, shuffle=True, random_state=9)

#%% Network

def make_nn():
    rb = l2(0.01)
    nn = Sequential()
    nn.add(Dense(64, activation='relu', kernel_regularizer=rb, bias_regularizer=rb, input_dim=13))
    nn.add(Dense(64, activation='relu', kernel_regularizer=rb, bias_regularizer=rb))
    nn.add(Dense(1, kernel_regularizer=rb, bias_regularizer=rb))
    nn.compile(loss='mean_squared_error', optimizer="Adagrad")
    return(nn)

r2_train_nn = list()
r2_test_nn  = list()
history     = list()
for train, test in kf_cv.split(X, y):
    nn = make_nn()
    nn.fit(X[train], y[train], batch_size=256, epochs=1000, verbose=0)
    history.append(nn.history.history["loss"])
    r2_train = r2_score(y[train], nn.predict(X[train]))
    r2_test  = r2_score(y[test],  nn.predict(X[test]))
    print('R2 train: %.4f, R2_test: %.4f' %(r2_train, r2_test))
    r2_train_nn.append(r2_train)
    r2_test_nn.append(r2_test)

print('Mean R2 train: %.4f (%.4f), Mean R2_test: %.4f (%.4f)' %(np.mean(r2_train_nn), np.std(r2_train_nn), np.mean(r2_test_nn), np.std(r2_test_nn)))

plt.plot(nn.history.history["loss"][950:])

#%% Linear model

r2_train_lm = list()
r2_test_lm  = list()
for train, test in kf_cv.split(X, y):
    lm = LinearRegression()
    lm.fit(X[train], y[train])
    r2_train = r2_score(y[train], lm.predict(X[train]))
    r2_test  = r2_score(y[test],  lm.predict(X[test]))
    print('R2 train: %.4f, R2_test: %.4f' %(r2_train, r2_test))
    r2_train_lm.append(r2_train)
    r2_test_lm.append(r2_test)

print('Mean R2 train: %.4f (%.4f), Mean R2_test: %.4f (%.4f)' %(np.mean(r2_train_lm), np.std(r2_train_lm), np.mean(r2_test_lm), np.std(r2_test_lm)))
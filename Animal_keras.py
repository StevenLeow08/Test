# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:25:00 2020

@author: Steven
"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# importing Data
X = pd.read_csv('Features.csv')
y = pd.read_csv('Targets.csv')

X.set_index('Animal', inplace=True)
y.set_index('Animal', inplace=True)

X = X.values
y = y.values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras import backend as K
def root_mean_squared_error(y_train, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_train))) 

"""from sklearn.metrics import mean_squared_error
from math import sqrt
def root_mean_squared_error(y_train, y_pred):
    return sqrt(mean_squared_error(y_train, y_pred))"""

#Building the Network 
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu', input_dim = 16))

# Adding the second hidden layer
regressor.add(Dense(output_dim = 34, init = 'uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = root_mean_squared_error, metrics = ['accuracy'])

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 10, nb_epoch = 300)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#!/usr/bin/env python3

import os
import sys

from itertools import product
import datetime

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from data_preparation import prepare_data


tf.config.experimental.set_visible_devices([], 'GPU') # Enforcing the usage of the CPU


# ---------------------------------------------------------------------------------------------------------------
# - Loading the data
if os.path.exists("data.csv"):
    data = pd.read_csv("data.csv")

else:
    data = prepare_data()

X  = data.drop(['score_home','score_away', 'winners'], axis = 1, inplace = False)
y = data.loc[:, ['score_home','score_away', 'winners']] 


# - Preprocessing

# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))

col_names_X = X.columns.tolist()

X = pd.DataFrame(scaler.fit_transform(X))
X.columns = col_names_X


# - Splitting in traning and test set
X_train, X_test, y_train, y_test = train_test_split(X,y)



# ---------------------------------------------------------------------------------------------------------------
# - Callbacks

# Tensorboard callback

current_path = os.path.dirname(os.path.realpath(__file__))
now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S").strip(" ").replace("/", "_").replace(":", "_").replace(" ", "_")

# log_directory = current_path + "/tensorboard/"+ dt_string

# tb_callback = TensorBoard(log_dir = log_directory)


# Learning rate scheduler

def generate_lr_scheduler(initial_learning_rate = 0.01):

    def lr_step_decay(epoch, lr):
        drop_rate = 0.5
        epochs_drop = 10.0
        return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

    return LearningRateScheduler(lr_step_decay)


# ---------------------------------------------------------------------------------------------------------------
# - Building the TF Model

def build_model(neurons = 8, depth = 3, activation = "relu", optimizer = "RMSprop", learning_rate = 0.01):
    """ GridSearch compatible model definition
    """
    
    # ---------------------------------------------------------------------------------------------------------------
    # - Callbacks
    
    # Tensorboard
    log_directory = current_path + "/tensorboard/"+ \
                    "_neu_" + str(neurons) + \
                    "_dep_" + str(depth) + \
                    "_act_" + str(activation) + \
                    "_opt_" + str(optimizer)

    tb_callback = TensorBoard(log_dir = log_directory)


    # Learning rate scheduler
    lr_callback = generate_lr_scheduler(initial_learning_rate= learning_rate)


    # ---------------------------------------------------------------------------------------------------------------
    # - Model definition
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Dense(neurons, activation=activation, input_shape=[X.shape[1]]))

    # Hidden layers
    for _ in range(depth):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))

    # Output layer
    model.add(tf.keras.layers.Dense(2))

    model.compile(
            loss = "mse",
            metrics = "mean_squared_error",
            optimizer=optimizer)

    model.fit(
            X_train,
            y_train.loc[:,['score_home', 'score_away']],
            epochs=50,
            batch_size=8,
            validation_split = 0.3,
            verbose=0,
            callbacks=[tb_callback, lr_callback])


    # ---------------------------------------------------------------------------------------------------------------
    # - Evaluation

    # Evaluating the regression model (predicts nr of goals for each team) with regard to the winning team
    # To do this, from the predicted scores the winning team must be calculated. This is then compared with
    # the actual result of the match

    y_predict = np.array(model.predict(X_test))

    y_predict_home = y_predict[:,0] # Scores of the home team
    y_predict_away = y_predict[:,1] # Scores of the away team

    predicted_result = np.empty([len(y_predict_home),1], dtype = 'str')
    predicted_result[np.greater(y_predict_home,y_predict_away)] = 'H'              # Home team wins
    predicted_result[y_predict_home < y_predict_away] = 'A'                        # Away team wins
    predicted_result[np.round(y_predict_home) == np.round(y_predict_away)] = 'D'   # Draw

    real_result = y_test.loc[:, 'winners'].values.tolist()                          # Which team (H,A,D) has really won

    # - Evaluating the regressor by comparing the predicted and the real winner
    nr_games = predicted_result.shape[0]

    print('Prediction accuracy: ',sum(predicted_result.reshape(1,-1)[0] == real_result)/nr_games*100)



param_grid = {"depth":[2,4,8],
                "activation": ["relu", "tanh"],
                "neurons":[16,64,128],
                "learning_rate": [0.1, 0.01, 0.01]}

# List of 
combinations_of_params = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]

for parameter in combinations_of_params:

    build_model(optimizer = parameter["optimizer"],
                depth = parameter["depth"],
                activation = parameter["activation"],
                neurons = parameter["neurons"],
                learning_rate=parameter["learning_rate"])





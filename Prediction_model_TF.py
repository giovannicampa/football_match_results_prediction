#!/usr/bin/env python3

import os
import sys

import numpy as np
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from data_preparation import prepare_data


tf.config.experimental.set_visible_devices([], 'GPU')

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

log_directory = current_path + "/tensorboard/"+ dt_string


tb_callback = TensorBoard(log_dir = log_directory)



# Learning rate scheduler

initial_learning_rate = 0.01
def lr_step_decay(epoch, lr):
    drop_rate = 0.5
    epochs_drop = 10.0
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

lr_callback = LearningRateScheduler(lr_step_decay)


# ---------------------------------------------------------------------------------------------------------------
# - Building the TF Model

# model = tf.keras.layer.Sequential([
#     tf.keras.layers.Dense(8, activation="relu", input_shape=[X.shape[1]]),
#     tf.keras.layers.Dense(8, activation="relu"),
#     tf.keras.layers.Dense(8, activation="relu"),
#     tf.keras.layers.Dense(8, activation="relu"),
#     tf.keras.layers.Dense(2)
# ])

input_layer = Input(shape=[X.shape[1]])
x = Dense(units = 8)(input_layer)
x = Dense(units = 8)(x)
x = Dense(units = 8)(x)

x_1 = Dense(units = 4)(x)
out_1 = Dense(units = 1)(x_1)

x_2 = Dense(units = 4)(x)
out_2 = Dense(units = 1)(x_2)

model = Model(inputs = input_layer, outputs = [out_1, out_2])



optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_squared_error'])

history = model.fit(
        X_train,
        y_train.loc[:,['score_home', 'score_away']],
        epochs=40,
        batch_size=8,
        validation_split = 0.3,
        verbose=1,
        callbacks=[tb_callback, lr_callback])


# ---------------------------------------------------------------------------------------------------------------
# - Evaluation

# Evaluating the regression model (predicts nr of goals for each team) with regard to the winning team
# To do this, from the predicted scores the winning team must be calculated. This is then compared with
# the actual result of the match

y_predict = model.predict(X_test)

y_predict_home = y_predict[0]
y_predict_away = y_predict[1]
y_predict.columns = ['score_home', 'score_away']
predicted_winners = np.empty([y_predict.shape[0],1], dtype = 'str')
predicted_winners[y_predict.score_home > y_predict.score_away] = 'H'
predicted_winners[y_predict.score_home < y_predict.score_away] = 'A'
predicted_winners[np.round(y_predict.score_home) == np.round(y_predict.score_away)] = 'D'

real_winners = y_test.loc[:, 'winners'].values.tolist()


# - Evaluating the regressor by comparing the predicted and the real winner
nr_games = predicted_winners.shape[0]

print('Prediction accuracy: ',sum(predicted_winners.reshape(1,-1)[0] == real_winners)/nr_games*100)

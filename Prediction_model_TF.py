import os
import sys

import numpy as np
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from data_preparation import prepare_data


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


tb_callback = TensorBoard(log_directory = log_directory)



class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir=log_directory, **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

tb_custom_callback = TrainValTensorBoard(log_dir=log_directory)


# Learning rate scheduler

initial_learning_rate = 0.01
def lr_step_decay(epoch, lr):
    drop_rate = 0.5
    epochs_drop = 10.0
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

lr_callback = LearningRateScheduler(lr_step_decay)


# ---------------------------------------------------------------------------------------------------------------
# - Building the TF Model

model = keras.Sequential([
    layers.Dense(8, activation=tf.nn.relu, input_shape=[X.shape[1]]),
    layers.Dense(8, activation=tf.nn.relu),
    layers.Dense(8, activation=tf.nn.relu),
    layers.Dense(8, activation=tf.nn.relu),
    layers.Dense(2)
])

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
        verbose=0,
        callbacks=[tb_callback, tb_custom_callback, lr_callback])


# ---------------------------------------------------------------------------------------------------------------
# - Evaluation

# Evaluating the regression model (predicts nr of goals for each team) with regard to the winning team
# To do this, from the predicted scores the winning team must be calculated. This is then compared with
# the actual result of the match

y_predict = pd.DataFrame(model.predict(X_test))
y_predict.columns = ['score_home', 'score_away']
predicted_winners = np.empty([y_predict.shape[0],1], dtype = 'str')
predicted_winners[y_predict.score_home > y_predict.score_away] = 'H'
predicted_winners[y_predict.score_home < y_predict.score_away] = 'A'
predicted_winners[np.round(y_predict.score_home) == np.round(y_predict.score_away)] = 'D'

real_winners = y_test.loc[:, 'winners'].values.tolist()


# - Evaluating the regressor by comparing the predicted and the real winner
nr_games = predicted_winners.shape[0]

print('Prediction accuracy: ',sum(predicted_winners.reshape(1,-1)[0] == real_winners)/nr_games*100)

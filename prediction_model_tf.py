#!/usr/bin/env python3

import os
import sys
import shutil

from itertools import product
import datetime

import numpy as np
import pandas as pd
import math

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

from data_preparation import prepare_data


tf.config.experimental.set_visible_devices([], 'GPU') # Enforcing the usage of the CPU
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ---------------------------------------------------------------------------------------------------------------
# - Loading the data
if os.path.exists("data.csv"):
    data = pd.read_csv("data.csv")

else:
    data = prepare_data()

X = data.drop(['score_home','score_away', 'winners'], axis = 1, inplace = False)
y = data.loc[:, ['score_home','score_away', 'winners']] 


# - Preprocessing

# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))

col_names_X = X.columns.tolist()

X = pd.DataFrame(scaler.fit_transform(X))
X.columns = col_names_X


# - Splitting in traning and test set
X_train, X_test, y_train, y_test = train_test_split(X,y)


def integer_accuracy(y_true, y_predict):

    accuracy = y_true - tf.keras.backend.round(y_predict)

    return accuracy

# ---------------------------------------------------------------------------------------------------------------
# - Callbacks

# Early stopping callback
es_callback = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0 ,restore_best_weights= True)

# Learning rate scheduler

def generate_lr_scheduler(initial_learning_rate = 0.01):

    def lr_step_decay(epoch, lr):
        drop_rate = 0.5
        epochs_drop = 10.0
        return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

    return LearningRateScheduler(lr_step_decay)


# ---------------------------------------------------------------------------------------------------------------
# - Building the TF Model
class ModelBuilder():

    def __init__(self, keep_n_best = 2):
        self.best_loss = 100
        self.best_n_losses = np.linspace(1000, 999, keep_n_best)
        self.best_n_losses_dirs = ["" for i in range(keep_n_best)]

        self.best_accuracy = 0
        self.best_n_accuracies = np.linspace(0, 0.01, keep_n_best)
        self.best_n_accuracies_dirs = ["" for i in range(keep_n_best)]

    def build(self, neurons = 8, depth = 3, activation = "relu", optimizer = "RMSprop", learning_rate = 0.01, batch_size = 8):
        """ Model definition with input parameters
        
        GridSearch compatible
        """
        
        # ---------------------------------------------------------------------------------------------------------------
        # - Callbacks
        
        # Tensorboard
        self.hp_string = "_nr_" + str(neurons) + \
                         "_depth_" + str(depth) + \
                         "_act_" + str(activation) + \
                         "_batch_"+str(batch_size) + \
                         "_lr_" + str(learning_rate)

        current_path = os.path.dirname(os.path.realpath(__file__))
        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S").strip(" ").replace("/", "_").replace(":", "_").replace(" ", "_")
        self.log_directory = current_path + "/tensorboard/"+ self.hp_string

        print(f"\nCurrently evaluating {self.hp_string}", end = "\r")

        tb_callback = TensorBoard(log_dir = self.log_directory, profile_batch=0)


        # Learning rate scheduler
        lr_callback = generate_lr_scheduler(initial_learning_rate= learning_rate)


        # ---------------------------------------------------------------------------------------------------------------
        # - Model definition
        self.model = tf.keras.models.Sequential()

        # Input layer
        self.model.add(tf.keras.layers.Dense(neurons, activation=activation, input_shape=[X.shape[1]]))

        # Hidden layers
        for _ in range(depth):
            self.model.add(tf.keras.layers.Dense(neurons, activation=activation))

        # Output layer
        self.model.add(tf.keras.layers.Dense(2))

        self.model.compile(
                loss = "mse",
                metrics = integer_accuracy,
                optimizer=optimizer)

        history = self.model.fit(
                X_train,
                y_train.loc[:,['score_home', 'score_away']],
                epochs=150,
                batch_size=batch_size,
                validation_split = 0.3,
                verbose=0,
                callbacks=[tb_callback, lr_callback, es_callback])

        self.current_loss = np.min(history.history["loss"])


    def evaluate_model(self):
        """Evaluating the regression model

        predicts nr of goals for each team with regard to the winning team
        To do this, from the predicted scores the winning team must be calculated.
        This is then compared with the actual result of the match
        """

        y_predict = np.array(self.model.predict(X_test))

        y_predict_home = y_predict[:,0] # Scores of the home team
        y_predict_away = y_predict[:,1] # Scores of the away team

        self.predicted_result = np.empty([len(y_predict_home),1], dtype = 'str')
        self.predicted_result[np.greater(y_predict_home,y_predict_away)] = 'H'              # Home team wins
        self.predicted_result[y_predict_home < y_predict_away] = 'A'                        # Away team wins
        self.predicted_result[np.round(y_predict_home) == np.round(y_predict_away)] = 'D'   # Draw

        self.real_result = y_test.loc[:, 'winners'].values.tolist()                          # Which team (H,A,D) has really won

        # - Evaluating the regressor by comparing the predicted and the real winner
        nr_games = self.predicted_result.shape[0]

        self.current_accuracy = accuracy_score(self.real_result, self.predicted_result)*100

        if self.current_accuracy > self.best_accuracy:
            self.best_accuracy = self.current_accuracy

            print("\x1b[6;30;42m" + f'Best accuracy ({self.best_accuracy}) is for {self.hp_string}. Loss: {self.current_loss}'+ "\x1b[0m")

            self.plot_confusion_matrix()


    def clean_tensorboard(self):
        """ Automatically removes uninteresting TB logs
        """

        id_worst_accuracy = np.argmin(self.best_n_accuracies)

        if self.current_accuracy > np.min(self.best_n_accuracies):
            
            if self.best_n_accuracies_dirs[id_worst_accuracy] != "":
                shutil.rmtree(self.best_n_accuracies_dirs[id_worst_accuracy])       # Remove the tb log of the worst model

            self.best_n_accuracies[id_worst_accuracy] = self.current_accuracy
            self.best_n_accuracies_dirs[id_worst_accuracy] = self.log_directory     # Replace the directory string of the worst model        

        # If the current model is not better than an existing one,
        # delete the current model's log
        elif self.best_n_accuracies_dirs[id_worst_accuracy] != "":

            shutil.rmtree(self.log_directory)


    def plot_confusion_matrix(self):

        fig, ax = plt.subplots()

        confusion_mat = confusion_matrix(
                y_true = self.real_result,
                y_pred = self.predicted_result.ravel().tolist(),
                normalize="true",
                labels = ["H", "D", "A"])

        im = ax.imshow(confusion_mat)

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])

        real_result = np.array(self.real_result)
        real_H_wins = sum(real_result == "H")
        real_D = sum(real_result == "D")
        real_A_wins = sum(real_result == "A")

        predicted_result = self.predicted_result.ravel()
        predicted_H_wins = sum(predicted_result == "H")
        predicted_D = sum(predicted_result == "D")
        predicted_A_wins = sum(predicted_result == "A")


        ax.set_xticklabels([f"Home wins ({real_H_wins})", f"Draw ({real_D})", f"Away wins ({real_A_wins})"])
        ax.set_yticklabels([f"Home wins ({predicted_H_wins})", f"Draw ({predicted_D})", f"Away wins ({predicted_A_wins})"])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        for row in range(confusion_mat.shape[0]):
            for col in range(confusion_mat.shape[1]):
                text = ax.text(
                        x = col,
                        y = row,
                        s = f"{confusion_mat[row, col]:.2f}",
                        ha="center",
                        va="center",
                        color="w")
        
        ax.set_xlabel(f"Predicted label", fontsize = 18)
        ax.set_ylabel(f"True label", fontsize = 18)

        ax.set_title(f"Prediction accuracy: {self.current_accuracy:.0f}", fontsize = 18)
        fig.tight_layout()
        # plt.show()
        plt.savefig("confusion_matrix.png")


param_grid = {"depth":[2,4,8],
                "activation": ["relu", "tanh"],
                "neurons":[8,16,32,64],
                "learning_rate": [0.1, 0.01],
                "batch_size":[32,64]}

# List of parameter combinations
combinations_of_params = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]


model_builder = ModelBuilder(keep_n_best=3)

for parameter in combinations_of_params:

    model_builder.build(depth = parameter["depth"],
                        activation = parameter["activation"],
                        neurons = parameter["neurons"],
                        learning_rate=parameter["learning_rate"],
                        batch_size=parameter["batch_size"])
    
    model_builder.evaluate_model()
    model_builder.clean_tensorboard()




import pickle
import os
from data_preparation import prepare_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

# Loading the prepared data
if os.path.exists("data.csv"):
    data = pd.read_csv("data.csv")

else:
    data = prepare_data()


# -------- First Part: Training and evaluating a RF regressor
    
# - 1.1: Performing a grid search for the best parameters of the random forest regressor
# - 1.2: Train the regressor on the whole feature set with the best parameters
# - 1.3: Evaluate the regressor by calculating the outcome of the game (won home, away, equal result) and compare it with the real result from y_test

X  = data.drop(['score_home','score_away', 'winners'], axis = 1, inplace = False)
y = data.loc[:, ['score_home','score_away', 'winners']] 

random_forest = RandomForestRegressor(n_jobs = -1)
neural_network = MLPRegressor(activation = 'relu', solver = "adam", early_stopping= True)

param_grid_rf = {'max_depth':list(range(3,10,2)), 'max_features': list(range(5,X.shape[1])), 'n_estimators': list(range(70,200,10))}
param_grid_mlp = {'hidden_layer_sizes': list(range(8,128,32))}


def find_best_params(regressor, param_grid, X, y):
    
    # Scaling X for better results
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    
    y = y.drop('winners', axis = 1)
    custom_scorer = make_scorer(mean_squared_error)
    regressor = RandomizedSearchCV(regressor, param_grid, cv = 3, scoring = custom_scorer, n_iter=10)
    regressor.fit(X,y)

#    print(np.sqrt(regressor.best_score_))
#    print(regressor.best_params_)
    return regressor.best_estimator_


# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))

col_names_X = X.columns.tolist()

X = pd.DataFrame(scaler.fit_transform(X))
X.columns = col_names_X

# Splitting in traning and test set
X_train, X_test, y_train, y_test = train_test_split(X,y)


# - 1.1: Find the best parameters for the regressors
regressor_rf = find_best_params(random_forest, param_grid_rf, X, y)

regressor_rf.fit(X_train, y_train.drop('winners', axis = 1))


# - 1.2 Instead of prediction the outcome as Home wins, Away wins.., we predict the scores and then we calculate the winner

y_predict = pd.DataFrame(regressor_rf.predict(X_test))
y_predict.columns = ['score_home', 'score_away']

predicted_winners = np.empty([y_predict.shape[0],1], dtype = 'str')
predicted_winners[np.round(y_predict.score_home) > np.round(y_predict.score_away)] = 'H'
predicted_winners[np.round(y_predict.score_home) < np.round(y_predict.score_away)] = 'A'
predicted_winners[np.round(y_predict.score_home) == np.round(y_predict.score_away)] = 'D'

real_winners = y_test.loc[:, 'winners'].values.tolist()


# - 1.3 Evaluating the regressor by looking comparing the predicted and the real winner
nr_games = predicted_winners.shape[0]

print('Prediction accuracy with all features: ',sum(predicted_winners.reshape(1,-1)[0] == real_winners)/nr_games*100)




# -------- Second Part: Train a MLP Regressor with the information about the most important featurs that we got from the RF regressor

# - 2.1: Looking for the best features to keep by looking at the features importance parameter of the RF regressor
# - 2.2: Remove the not important features from the feature set
# - 2.3: Look for the best parameters of RF regressor trained with the reduced training set
# - 2.4: Evaluate the regressor with the reduced features


range_feature_importances = np.linspace(0.01, 0.2, 5)

for feat_importance in range_feature_importances:
    # - 2.1: Finding the most important features and retraining the model on them only
    feature_relevance = (regressor_rf.feature_importances_>feat_importance)
    feature_names = X.columns[feature_relevance]


    # - 2.2: Remove the not important features from the feature set
    X_reduced = X.loc[:, feature_names]

    # Splitting in training and test sets
    Xred_train, Xred_test, y_train, y_test = train_test_split(X_reduced, y)


    # - 2.3: Find the best parameters for the regressors
    regressor_mlp = find_best_params(neural_network, param_grid_mlp, X_reduced, y)
    regressor_mlp.fit(Xred_train, y_train.drop('winners', axis = 1))


    # - 2.4: Instead of prediction the outcome as Home wins, Away wins.., we predict the scores and then we calculate the winner

    y_predict = pd.DataFrame(regressor_mlp.predict(Xred_test))
    y_predict.columns = ['score_home', 'score_away']

    predicted_winners = np.empty([y_predict.shape[0],1], dtype = 'str')
    predicted_winners[np.round(y_predict.score_home) > np.round(y_predict.score_away)] = 'H'
    predicted_winners[np.round(y_predict.score_home) < np.round(y_predict.score_away)] = 'A'
    predicted_winners[np.round(y_predict.score_home) == np.round(y_predict.score_away)] = 'D'

    real_winners = y_test.loc[:, 'winners'].values.tolist()


    nr_games = predicted_winners.shape[0]

    prediction_accuracy = accuracy_score(predicted_winners.reshape(1,-1)[0], real_winners)

    print(f'Prediction accuracy with most relevant features: {prediction_accuracy:.2f} with features: {feature_names}')
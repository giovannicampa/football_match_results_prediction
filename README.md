# Football match results predictor
This project uses a machine learning approach to predict the number of goals scored by two teams in a match and then calculates the winning team.

## Project structure
The contained files are:


### data_preparation.py

Reads the data from the csv files containing the information about every single football match of various the seasons.
It then calculates features such as ranking position of the two teams at the moment of the match, average values of the scores, yellow cards and others
It is called by Prediction_mdel.py and only executed if the prepared data has not been already pickled

 
### prediction_model.py

Predicts the numeric value of goals scored by the two teams. It is divided in two:

1 - A random forest regressor is trained on the dataset, the best parameters found with a grid search and in the end the outcome is translated to winner home team,
    winner away team or same score. The accuracy of the outcome is then evaluated by comparing the predicted outcome to the real one

2 - Here the most important features indicated by the random forest regressor are found and this information is then used to reduce the feature set.
The best parameters for this algorithm are again found with a grid search. The final model, with the best parameters and the most important features is then trained and evaluated on the final result as winner home team, away team or same score.
    

### prediction_model_TF.py

A multi-output neural network build with tensorflow's the Keras API is used to predict the result of the match by predicting the score of each team.

#### Parameter search
To find the best hyperparameters, a **grid search** has been used. The parameters looked for are network shape (**depth** and **layer size**), **activation function**, **batch size** and **learning rate**.

#### Custom accuracy score
As the score prediction is a regression problem and does not consider the discrete nature of the match result, a custom accuracy score has been defined (here **integer_accuracy**). This metric compares the rounded predicted value with the real one (integer).

#### Other callbacks
Besides the tensorboard callback, also a **learning rate scheduler** and an **early stopping** callback have been used.

<figure>
  <img src="https://github.com/giovannicampa/football_match_results_prediction/blob/master/pictures/tb_logs" width="500">
  <figcaption>Tensorboard log for loss and custom accuracy score</figcaption>
</figure>

## Results
The results of the models have to be evaluated by considering the fact that a random guess has an accuracy of 33% (home team wins, away team wins, draw).

Both the classical machine learning and the deep learning model achieve better accuracies than the random guess.

**Random forest**  with optimal parameters has an accuracy of 50%

**MLP** (sklearn) with optimal parameters has an accuracy of 55%

**TF model** with optimal parameters has an accuracy of 57%

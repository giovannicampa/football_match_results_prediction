# Football match results predictor
This project uses a machine learning approach to predict the number of goals scored by two teams in a match and then calculates the winning team.

## Project structure
The contained files are:

- **data_preparation.py**

Reads the data from the csv files containing the information about every single football match of various the seasons.
It then calculates features such as ranking position of the two teams at the moment of the match, average values of the scores, yellow cards and others
It is called by Prediction_mdel.py and only executed if the prepared data has not been already pickled
  
 
- **Prediction_model.py**

Predicts the numeric value of goals scored by the two teams. It is divided in two:

1 - A random forest regressor is trained on the dataset, the best parameters found with a grid search and in the end the outcome is translated to winner home team,
    winner away team or same score. The accuracy of the outcome is then evaluated by comparing the predicted outcome to the real one
    
2 - Here the most important features indicated by the random forest regressor are found and this information is then used to reduce the feature set.
The best parameters for this algorithm are again found with a grid search. The final model, with the best parameters and the most important features is then trained and evaluated on the final result as winner home team, away team or same score.
    
- **Prediction_model_TF.py**

A multi-output neural network is used to predict the result of the match by predicting the score of each team.


## Results
Both models can predict the outcome (winner home team, away team or same score) with a higher accuracy than a random guess (33%).
Random forest regressor with optimal parameters has an accuracy on the result of about 50% (17 percentage points improvement on random guesses)
MLP regressor with optimal parameters has an accuracy on the result of about 55% (23 percentage points improvement on random guesses)

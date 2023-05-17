# Imports

# For modules
import os
import sys

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

# Data Analysis
import pandas as pd


# Metrics
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

def modeling(all_models, X_train, y_train, X_valid, y_valid, X_test, y_test, note=''):
    
    result_df = pd.DataFrame()

    for single_model in modeling:

        # Get info
        name = single_model['name']
        model = single_model['model']

        # Fit the model
        model.fit(X_train, y_train)

        # Get Predictions
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        y_test_pred = model.predict(X_test)

        # Get training scores
        # Calculate Accuracy, F1, and MCC Values of the training set
        accuracy_train = accuracy_score(y_train, y_train_pred)*100
        f1_positive_train = f1_score(y_train, y_train_pred)*100
        f1_negative_train = f1_score(1-y_train, 1-y_train_pred)*100
        mcc_train = matthews_corrcoef(y_train, y_train_pred)*100

        # Calculate Accuracy, F1, and MCC Values of the Validation set
        accuracy_valid = accuracy_score(y_valid, y_valid_pred)*100
        f1_positive_valid = f1_score(y_valid, y_valid_pred)*100
        f1_negative_valid = f1_score(1-y_valid, 1-y_valid_pred)*100
        mcc_valid = matthews_corrcoef(y_valid, y_valid_pred)*100

        # Calculate Accuracy, F1, and MCC Values of the testing set
        accuracy_test = accuracy_score(y_test, y_test_pred)*100
        f1_positive_test = f1_score(y_test, y_test_pred)*100
        f1_negative_test = f1_score(1-y_test, 1-y_test_pred)*100
        mcc_test = matthews_corrcoef(y_test, y_test_pred)*100



    return















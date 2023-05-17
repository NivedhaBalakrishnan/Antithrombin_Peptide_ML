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

# XGB
from xgboost import XGBClassifier

# K-Fold
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# Metrics
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# File
import json

# Fucnction for K-fold
def cv_fold(all_models, X_training, y_training, X_test, y_test, fold=5, note=''):
    """
    This function performs k-fold cross-validation on a list of models, and returns the mean and standard deviation of the performance metrics 
    on the training, validation and test sets for each model.
    
    Parameters:
    all_models (list): A list of dictionaries where each dictionary contains the model object and the model name
    X_training (pandas.DataFrame): The feature matrix for the training set
    y_training (pandas.Series): The target variable for the training set
    X_test (pandas.DataFrame): The feature matrix for the test set
    y_test (pandas.Series): The target variable for the test set
    fold (int): The number of folds to use in the cross-validation process (default is 5)
    
    Returns:
    allKFold (pandas.DataFrame): A dataframe containing the mean and standard deviation of the performance metrics on the training, 
    validation and test sets for each model
    """

    # Empty dataframe to save all results
    allKFold = pd.DataFrame()
    
    # loop through all models
    for single_model in all_models:

        X_training_temp = X_training.copy()
        X_test_temp = X_test.copy()

        if note == 'final':
            with open(parent_dir+'/dependency/features/best_features_'+single_model['name']+'.json') as json_file:
                features = json.load(json_file)

            X_training_temp = X_training_temp[features]
            X_test_temp = X_test_temp[features]

        # for XGB change weight depending upon imbalance
        if single_model['name'] == 'XGB':
            value_counts = y_training.value_counts()
            weight = value_counts[0]/value_counts[1]
            single_model['model'] = XGBClassifier(scale_pos_weight=weight)

        # Assign the model and train the data
        model = single_model['model']
        
        # Call StratifiedKFold function to split the data
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        
        # Creating empty dictionary to store 10 batches of performance metrics
        tempDf = pd.DataFrame()
        
        # Using for loop to loop through 10 batches
        for train_index, test_index in skf.split(X_training_temp,y_training):
            # Generate train and test sets
            X_train_kfold, X_valid_kfold = X_training_temp.iloc[train_index], X_training_temp.iloc[test_index]
            y_train_kfold, y_valid_kfold = y_training.iloc[train_index], y_training.iloc[test_index]
            
            smote = SMOTE(random_state=42)
            X_train_kfold, y_train_kfold = smote.fit_resample(X_train_kfold, y_train_kfold)

            # Model Fitting
            model.fit(X_train_kfold,y_train_kfold)
            
            # Now we make the predictions on both the training and test sets of the model.
            y_pred_train_kfold = model.predict(X_train_kfold)
            y_pred_valid_kfold = model.predict(X_valid_kfold)
            y_pred_test = model.predict(X_test_temp)
            
            # Calculate Accuracy, F1, and MCC Values of the Validation set
            accuracy_valid = accuracy_score(y_valid_kfold, y_pred_valid_kfold)*100
            f1_positive_valid = f1_score(y_valid_kfold, y_pred_valid_kfold)*100
            f1_negative_valid = f1_score(1-y_valid_kfold, 1-y_pred_valid_kfold)*100
            mcc_valid = matthews_corrcoef(y_valid_kfold, y_pred_valid_kfold)*100

            # Calculate Accuracy, F1, and MCC Values of the training set
            accuracy_train = accuracy_score(y_train_kfold, y_pred_train_kfold)*100
            f1_positive_train = f1_score(y_train_kfold, y_pred_train_kfold)*100
            f1_negative_train = f1_score(1-y_train_kfold, 1-y_pred_train_kfold)*100
            mcc_train = matthews_corrcoef(y_train_kfold, y_pred_train_kfold)*100
            
            # Calculate Accuracy, F1, and MCC Values of the testing set
            accuracy_test = accuracy_score(y_test, y_pred_test)*100
            f1_positive_test = f1_score(y_test, y_pred_test)*100
            f1_negative_test = f1_score(1-y_test, 1-y_pred_test)*100
            mcc_test = matthews_corrcoef(y_test, y_pred_test)*100
            
            # create a dictionary and dataframe to store the metrics of the current batch
            batchDict = {'Train Accuracy':accuracy_train,'Train F1 Negative' : f1_negative_train, 'Train F1 Positive':f1_positive_train, 'Train MCC Score' : mcc_train, 
                    'Validation Accuracy':accuracy_valid,'Validation F1 Negative' : f1_negative_valid, 'Validation F1 Positive':f1_positive_valid, 'Validation MCC Score' : mcc_valid,
                        'Test Accuracy':accuracy_test,'Test F1 Negative' : f1_negative_test, 'Test F1 Positive':f1_positive_test, 'Test MCC Score' : mcc_test}
            batchDf = pd.DataFrame(batchDict, index=[0])
            
            # Concatenate the current batch metrics to temp dataframe to store all 10 batches metrics
            tempDf = pd.concat([tempDf,batchDf], ignore_index=True)
        
        # Mean values of the Validation set
        accuracy_mean_valid = tempDf['Validation Accuracy'].mean(axis=0)
        f1_positive_mean_valid = tempDf['Validation F1 Positive'].mean(axis=0)
        f1_negative_mean_valid = tempDf['Validation F1 Negative'].mean(axis=0)
        mcc_mean_valid = tempDf['Validation MCC Score'].mean(axis=0)

        # Mean values of the train set
        accuracy_mean_train = tempDf['Train Accuracy'].mean(axis=0)
        f1_positive_mean_train = tempDf['Train F1 Positive'].mean(axis=0)
        f1_negative_mean_train = tempDf['Train F1 Negative'].mean(axis=0)
        mcc_mean_train = tempDf['Train MCC Score'].mean(axis=0)
        
        # Mean values of the test set
        accuracy_mean_test = tempDf['Test Accuracy'].mean(axis=0)
        f1_positive_mean_test = tempDf['Test F1 Positive'].mean(axis=0)
        f1_negative_mean_test = tempDf['Test F1 Negative'].mean(axis=0)
        mcc_mean_test = tempDf['Test MCC Score'].mean(axis=0)
        
        # Std values of the Validation set
        mcc_std_valid = tempDf['Validation MCC Score'].std(axis=0)

        # Std values of the train set
        mcc_std_train = tempDf['Train MCC Score'].std(axis=0)
        
        # Std values of the test set
        mcc_std_test = tempDf['Test MCC Score'].std(axis=0)
        
        # Collect the means in the dataframe
        currentKFold = {'Name':single_model['name'], 'Train Accuracy':accuracy_mean_train,
                        'Train F1 Negative' : f1_negative_mean_train, 'Train F1 Positive':f1_positive_mean_train, 
                        'Train MCC Score' : mcc_mean_train, 'Train MCC Std':mcc_std_train,
                        'Validation Accuracy':accuracy_mean_valid,
                        'Validation F1 Negative' : f1_negative_mean_valid, 
                        'Validation F1 Positive':f1_positive_mean_valid, 'Validation MCC Score' : mcc_mean_valid,
                        'Validation MCC Std':mcc_std_valid,
                    'Test Accuracy':accuracy_mean_test,
                        'Test F1 Negative' : f1_negative_mean_test, 'Test F1 Positive':f1_positive_mean_test, 
                        'Test MCC Score' : mcc_mean_test, 'Test MCC Std':mcc_std_test}
        
        currentKFold = pd.DataFrame(currentKFold, index=[0])
        
        # Concatenate currentKFold dataframe to allKFold dataframe
        allKFold = pd.concat([allKFold, currentKFold], ignore_index=True)

    return allKFold

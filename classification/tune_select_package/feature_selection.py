# For modules
import os
import sys

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

# Imports
from kfold_package.kfold import cv_fold

# save & load
import json
import pickle

# Data Analysis
import pandas as pd

# RFE
from sklearn.feature_selection import RFE


# Get saved models
def get_models():
    
    # SVC RBF model
    with open(parent_dir+'/dependency/pickle/untrained/svcrbf_before_fs.pkl', 'rb') as f:
        svc_rbf_model = pickle.load(f)
    svc_rbf = {'name':'svcrbf', 'model':svc_rbf_model}

    # SVC Linear model
    with open(parent_dir+'/dependency/pickle/untrained/svclin_before_fs.pkl', 'rb') as f:
        svc_lin_model = pickle.load(f)
    svc_lin = {'name':'svclin', 'model':svc_lin_model}

    # Logistic model
    with open(parent_dir+'/dependency/pickle/untrained/logistic_before_fs.pkl', 'rb') as f:
        logistic_model = pickle.load(f)
    logistic = {'name':'logistic', 'model':logistic_model}

    # Random model
    with open(parent_dir+'/dependency/pickle/untrained/random_before_fs.pkl', 'rb') as f:
        random_model = pickle.load(f)
    random = {'name':'random', 'model':random_model}

    # KNN model
    with open(parent_dir+'/dependency/pickle/untrained/knn_before_fs.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    knn = {'name':'knn', 'model':knn_model}

    # XGB model
    with open(parent_dir+'/dependency/pickle/untrained/xgb_before_fs.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    xgb = {'name':'xgb', 'model':xgb_model}


    return svc_rbf, svc_lin, logistic, random, knn, xgb




# RFE function
def rfe_modelling (rfe_models, X_training, y_training, X_test, y_test):

    # To save best features
    feature_dict = {}

    # Get total number of features
    total = X_training.shape[1]

    for model_dict in rfe_models:
        print('\n\n\n')
        print(model_dict['name'])
        
        model = model_dict['model'] # get model
        
        all_features = {} # to save features in each iterations
        X_rfe = pd.DataFrame() # dataframe to save results
        
        print('Count Down')

        for feature in reversed(range(1,total,1)):
            print('In')
            # Count down
            print(feature, end=' ')
        
            rfe_model = RFE(model, n_features_to_select=feature, step=1)
            rfe_model.fit(X_training, y_training)
            
            X_selected = rfe_model.transform(X_training)
            X_selected_labels = rfe_model.get_support()
            X_selected_features = list(filter(None, X_training.columns * X_selected_labels))
        
            all_features[feature] = X_selected_features
        
            X_training = pd.DataFrame(data=X_selected,columns=X_selected_features)
            
            # Change dataframe of X_test
            X_test = X_test[X_training.columns.tolist()]

            X_kfold = cv_fold([model_dict], X_training, y_training, X_test, y_test)
            X_kfold['Num Features'] = feature
        
            X_rfe = pd.concat([X_rfe, X_kfold])

        print()
        # Save Results
        X_rfe.to_csv(parent_dir+'/results/feature selection/rfe_'+model_dict['name']+'.csv', index=False)  # RFE results
        with open(parent_dir+"/results/feature selection/features_"+model_dict['name']+".json", "w") as f:
            json.dump(all_features, f) # save features
        
        print(X_rfe)
        
        # Get best features
        X_rfe_best = X_rfe.sort_values(by=['Validation MCC Score', 'Num Features'], ascending=[False, True])
        best_row = X_rfe_best.iloc[0]
        
        # Get best Num Features
        num_features = best_row['Num Features'].astype(int)
        
        # Get the corresponding Train MCC Score and Test MCC Score
        train_mcc = best_row['Train MCC Score'].astype(float)
        valid_mcc = best_row['Validation MCC Score'].astype(float)
        test_mcc = best_row['Test MCC Score'].astype(float)

        # Print Results
        print(f"Model: {model_dict['name']}")
        print(f'Best Number of Features: {num_features}')
        print(f'\tTrain MCC: {train_mcc}')
        print(f'\tValid MCC: {valid_mcc}')
        print(f'\tTest MCC:  {test_mcc}\n')

        # Save best features in dependency folder
        best_features = all_features[num_features]
        with open(parent_dir+"/dependency/features/best_features_"+model_dict['name']+".json", "w") as f:
            json.dump(best_features, f) # save best features
    
        feature_dict[model_dict['name']] = best_features

    return feature_dict



# Feature Selection
def select_best_features(X_training, y_training, X_test, y_test, mode='rfe', all_models=None):
    
    # get trained models
    if all_models==None:
        svc_rbf, svc_lin, logistic, random, knn, xgb = get_models()        
    else:    
        svc_rbf, svc_lin, logistic, random, knn, xgb = all_models

    # Models with RFE
    rfe_models = [svc_lin, logistic, random]

    # Do RFE
    if (mode == 'rfe') or (mode == 'both'):
        rfe_feature_dict = rfe_modelling(rfe_models, X_training, y_training, X_test, y_test)

    return rfe_feature_dict


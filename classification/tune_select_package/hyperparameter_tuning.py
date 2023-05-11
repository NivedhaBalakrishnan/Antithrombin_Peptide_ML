# For modules
import os

# Get the path to the current directory
class_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Get the path to the parent directory
# parent_dir = os.path.dirname(current_dir)

# # Add the parent directory to the Python path
# sys.path.append(parent_dir)

# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef

# Data Analysis
import pandas as pd
import numpy as np

# ML Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# save model
import pickle
import json


# tuning any model
def tuning_model(X_training, y_training, model_params, k, iter, seed, note):
    # sanity check
#     print(model_params['name'], X_training.shape)

    # Unpack dictionary
    model = model_params['model']
    params = model_params['params']

    # Set k
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    # Initialize Randomized Search CV
    search = RandomizedSearchCV(model, params, n_iter=iter, scoring=make_scorer(matthews_corrcoef), cv=cv, random_state=seed, n_jobs=-1, return_train_score=True)
    search.fit(X_training, y_training)

    # Get best parameters and score
    best_params = search.best_params_  # dictionary of best hyperparameters
    best_test_mean = search.cv_results_['mean_test_score'][search.best_index_]  # test mean score of best hyperparameters
    best_train_mean = search.cv_results_['mean_train_score'][search.best_index_]  # train mean score of best hyperparameters

    # Print
    print(f"Best hyperparameter for {model_params['name']}")
    print(f'\tBest Parameters: {best_params}')
    print(f'\tTest Score: {best_test_mean}')
    print(f'\tTrain Score: {best_train_mean}\n')

    # Get Attributes
    attr = pd.json_normalize(search.cv_results_['params'])
    attr['rank'] = search.cv_results_['rank_test_score']
    attr['test_means'] = search.cv_results_['mean_test_score']
    attr['test_stds'] = search.cv_results_['std_test_score']
    attr['train_means'] = search.cv_results_['mean_train_score']
    attr['train_stds'] = search.cv_results_['std_train_score']

    # save the results
    attr.to_csv(class_dir+'/results/hypertune/'+model_params['name']+'_'+note+'.csv')

    # Best model
    best_model = model.set_params(**best_params)
    model_dict = {'name': model_params['name'], 'model': best_model}

    with open(class_dir+'/dependency/pickle/untrained/'+model_params['name']+'_'+note+'.pkl', 'wb') as f:
           pickle.dump(best_model, f)

    return model_dict

# Functions to get model parameters
def get_svcrbf_params():
        # SVC RBF kernel
        params = {'gamma' : np.arange(0.001, 5, 0.001), 'C' :  np.arange(0.0001, 5, 0.0001)}

        # Model dictionary
        svcrbf_params = {'name': 'svcrbf', 'model': SVC(class_weight='balanced'), 'params': params}

        return svcrbf_params

def get_svclin_params():
        # SVC Linear kernel
        params = {'gamma' : np.arange(0.001, 5, 0.001), 'C' :  np.arange(0.0001, 5, 0.0001)}

        # Model dictionary
        svclin_params = {'name': 'svclin', 'model': SVC(kernel='linear', class_weight='balanced'), 'params': params}

        return svclin_params

def get_logistic_params():
        # Logistic
        params = {'penalty' : ['l1', 'l2'], 'C' :  np.arange(0.001, 10, 0.001)}
        model = LogisticRegression(class_weight = 'balanced', solver = 'liblinear')
        
        # Model dictionary
        logistic_params = {'name':'logistic', 'model':model, 'params':params}

        return logistic_params

def get_random_params():
        # Random Forest
        n_estimators = list(range(10,500,10))
        max_depth = list(range(2,8,1))
        max_samples = np.arange(0.1,1.1,0.1)
        bootstrap = [True]

        params = {'n_estimators'   : n_estimators, 
                    'max_depth'    : max_depth,
                    'max_samples'  : max_samples,
                    'bootstrap'    : bootstrap}
    
        model = RandomForestClassifier(class_weight='balanced')
        
        # Model dictionary
        random_params = {'name':'random', 'model':model, 'params':params}

        return random_params

def get_knn_params():
        # KNN
        params = {'n_neighbors' : range(1,22,1), 'weights' : ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        model = KNeighborsClassifier()
        
        # Model dictionary
        knn_params = {'name':'knn', 'model':model, 'params':params}

        return knn_params

def get_xgb_params(y_training):
        # XGBoost
        params = {
        'learning_rate': np.linspace(0.1, 1, 10),    # learning rate
        'n_estimators': np.arange(100, 1000, 100),  # number of trees
        'max_depth': np.arange(3, 8),              # maximum depth of each tree
        'subsample': np.linspace(0.5, 1, 5),       # subsample ratio
        'colsample_bytree': np.linspace(0.5, 1, 5), # subsample ratio of columns
        'reg_alpha': np.logspace(-4, 1, 20),        # L1 regularization parameter
        'reg_lambda': np.logspace(-4, 1, 20)        # L2 regularization parameter
        }

        # Setting for Imabalanced dataset
        value_counts = y_training.value_counts()
        weight = value_counts[0]/value_counts[1]

        # Model dictionary
        xgb_params = {'name': 'xgb', 'model': XGBClassifier(scale_pos_weight=weight), 'params': params}

        return xgb_params



# Hyperparameter tuning of all models
def hypertune(X_training, y_training, k=5, iter=200, seed=33, note=''):
    
    
    # Get model parameters
    svcrbf_params = get_svcrbf_params()
    svclin_params = get_svclin_params()
    logistic_params = get_logistic_params()
    random_params = get_random_params()
    knn_params = get_knn_params()
    xgb_params = get_xgb_params(y_training)

#     if note == 'after_fs':
#         with open(class_dir+'/dependency/features/best_features_svclin.json') as f:
#                 svclin_features = json.load(f)
        
#         with open(class_dir+'/dependency/features/best_features_logistic.json') as f:
#                 logistic_features = json.load(f)
        
#         with open(class_dir+'/dependency/features/best_features_random.json') as f:
#                 random_features = json.load(f)
        
#         svc_lin = tuning_model(X_training[svclin_features], y_training, svclin_params, k, iter, seed, note)
#         logistic = tuning_model(X_training[logistic_features], y_training, logistic_params, k, iter, seed, note)
#         random = tuning_model(X_training[random_features], y_training, random_params, k, iter, seed, note)
    
#         all_models = [svc_lin, logistic, random]

    if note == 'after_fs':
        with open(class_dir+'/dependency/features/best_features_svcrbf.json') as f:
                svcrbf_features = json.load(f)
        
        with open(class_dir+'/dependency/features/best_features_knn.json') as f:
                knn_features = json.load(f)
        
        with open(class_dir+'/dependency/features/best_features_xgb.json') as f:
                xgb_features = json.load(f)
        
        svc_rbf = tuning_model(X_training[svcrbf_features], y_training, svcrbf_params, k, iter, seed, note)
        knn = tuning_model(X_training[knn_features], y_training, knn_params, k, iter, seed, note)
        xgb = tuning_model(X_training[xgb_features], y_training, xgb_params, k, iter, seed, note)
    
        all_models = [svc_rbf, knn, xgb]
    
    else:
    
        # Hyperparameter tuning
        svc_rbf = tuning_model(X_training, y_training, svcrbf_params, k, iter, seed, note)
        svc_lin = tuning_model(X_training, y_training, svclin_params, k, iter, seed, note)
        logistic = tuning_model(X_training, y_training, logistic_params, k, iter, seed, note)
        random = tuning_model(X_training, y_training, random_params, k, iter, seed, note)
        knn = tuning_model(X_training, y_training, knn_params, k, iter, seed, note)
        xgb = tuning_model(X_training, y_training, xgb_params, k, iter, seed, note)

        all_models = [svc_rbf, svc_lin, logistic, random, knn, xgb]

    return all_models
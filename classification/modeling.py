# For modules
import os
import sys

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

# Import
from data_preprocessing_packages.preprocessing import import_split_scale
from tune_select_package.hyperparameter_tuning import hypertune
from tune_select_package.feature_selection import select_best_features
from kfold_package.kfold import cv_fold
from kfold_package.plot_kfold import plot_kfold_mcc, plot_kfold_variation

# Data Analysis
import pandas as pd

# Plot
import matplotlib.pyplot as plt


# Function for hypertuning 
def hypertuning(X_training, y_training, X_test, y_test, k=5, iter=5, seed=33, note=''):
    # Get tuned model
    all_models = hypertune(X_training, y_training, k, iter, seed, note)

    # if note=='before_fs':
    #     # Do kfold with tuned models
    #     kfold = cv_fold(all_models, X_training, y_training, X_test, y_test)
    #     print(kfold) # print
    #     kfold.to_csv(current_dir+'/results/kfold/kfold_'+str(rs)+'_'+str(ss)+'_'+str(note)+'.csv', index=False) # save the results

    #     # Plot the results
    #     plot = plot_kfold_mcc(kfold) # call the function to get the plot
    #     plt.show() # display the plot
    #     plot.savefig(current_dir+'/figures/kfold/kfold_mcc'+str(rs)+'_'+str(ss)+'_'+str(note)+'.png') # save the plot

    return all_models


# -------------------------------- Essentials
# Parameters
rs=33 # Set random seed
ss=42 # Set shuffle seed

# Import, Split and Preprocess
X_training, y_training, X_train, y_train, X_valid, y_valid, X_test, y_test = import_split_scale(random_state=rs, shuffle=ss)


# -------------------------------- Before Feature Selection
# Parameters
note = 'before_fs' # Set note to save results
iter = 200 # Set number of iteration for RandomizedSearchCV 

# Set None if not running hypertuning
all_models = None

# Get tuned models
# all_models = hypertuning(X_training, y_training, X_test, y_test, k=5, iter=iter, seed=rs, note=note)

mode = 'sfs' # or 'rfe' or 'sfs'


# --------------------------------- Feature Selection
rfe_feature_dict = select_best_features(X_training, y_training, X_test, y_test, mode, all_models)


# --------------------------------- After Feature Selection
# Parameters
note = 'after_fs' # Set note to save results
iter = 200 # Set number of iteration for RandomizedSearchCV

# Get tuned models
all_models = hypertuning(X_training, y_training, X_test, y_test, k=5, iter=iter, seed=rs, note=note)
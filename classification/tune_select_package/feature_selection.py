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






def select_best_features(all_models, X_training, y_training, X_test, y_test):
    svc_rbf, svc_lin, logistic, random, knn, xgb = all_models

    # Models with RFE
    
    return


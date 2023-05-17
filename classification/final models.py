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

# Import module from the package
from data_preprocessing_packages.preprocessing import import_split_scale
from kfold_package.kfold import cv_fold
from kfold_package.plot_kfold import plot_kfold_mcc, plot_kfold_variation

# Data Analysis
import pandas as pd

# Import models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Plot
import matplotlib.pyplot as plt

# Model & files
import pickle
import json


def kfold_random_state(rs, ss, note=''):
    # Get split datasets
    X_training, y_training, X_train, y_train, X_valid, y_valid, X_test, y_test = import_split_scale(random_state=rs, shuffle=ss)

    """Call the functions of the selected ML algorithms
    The parameters of each model are selected through Randomized Search CV method"""

    # SVC Linear model
    with open(current_dir+'/dependency/pickle/untrained/svclin_before_fs.pkl', 'rb') as f:
        svc_lin_model = pickle.load(f)
    svc_lin = {'name':'svclin', 'model':svc_lin_model}

    
    # Logistic model
    with open(current_dir+'/dependency/pickle/untrained/logistic_before_fs.pkl', 'rb') as f:
        logistic_model = pickle.load(f)
    logistic = {'name':'logistic', 'model':logistic_model}

    # Random model
    with open(current_dir+'/dependency/pickle/untrained/random_before_fs.pkl', 'rb') as f:
        random_model = pickle.load(f)
    random = {'name':'random', 'model':random_model}

    # collect all the models in a list
    # all_models = [svm_rbf_model, svm_linear_model, logistic_model, random_model, knn_model, xgboost_model]
    all_models = [svc_lin, logistic, random]

    # do kfold
    kfold = cv_fold(all_models, X_training, y_training, X_test, y_test, note)
    print(kfold) # print
    kfold.to_csv(current_dir+'/results/kfold/kfold_3_'+str(rs)+'_'+str(ss)+'_'+str(note)+'.csv', index=False) # save the results

    # # Plot the results
    plot = plot_kfold_mcc(kfold) # call the function to get the plot
    plt.show() # display the plot
    plot.savefig(current_dir+'/figures/kfold/kfold_3_'+str(rs)+'_'+str(ss)+'_'+str(note)+'.png') # save the plot

    return


# Combine CSV
def combine_kfold():
    directory = os.path.dirname(os.path.abspath(__file__))+'/results'
    
    print(directory)

    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

    # Create an empty DataFrame to hold the combined data
    combined_data = pd.DataFrame()

    # Loop through each CSV file, read it into a DataFrame, and concatenate it to the combined data
    for file in csv_files:
        data = pd.read_csv(file)
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Write the combined data to a new CSV file
    # combined_data.to_csv(os.path.join(directory, 'combined_data.csv'), index=False)
    print(f'Number of samples: {len(combined_data)//6}')

    return combined_data


# Set random Seed
# random = [[53,33], [53,42], [53,72], [57, 65], [65, 57], [42, 33], [42, 42], [42, 72], [33, 72], [57, 72]]

# for i in random:
#     rs, ss = i

    # Class the kfold function
kfold_random_state(rs=33, ss=42, note='final_old_params')

# # Get combined data
# combined_data = combine_kfold()

# # Plot variations
# plot_kfold_variation(combined_data)
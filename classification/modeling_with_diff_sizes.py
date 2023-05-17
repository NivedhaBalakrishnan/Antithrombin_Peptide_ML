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
from data_preprocessing_packages.preprocessing_ratio import import_split_scale as imp_ratio
from data_preprocessing_packages.preprocessing import import_split_scale as imp
from tune_select_package.feature_selection import get_models
from kfold_package.kfold import cv_fold
from kfold_package.plot_kfold import plot_kfold_mcc, plot_kfold_variation

# Data Analysis
import pandas as pd

# # Import models
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier

# Plot
import matplotlib.pyplot as plt

# Files
import glob


# ------------------------------------------------------- kfold with ratio
# With different ratios
def kfold_ratio_model(rs, ss):

    # Get tuned models for all features
    svc_rbf, svc_lin, logistic, random, knn = get_models()
    all_models = [svc_rbf, svc_lin, logistic, random, knn]

    # negative ratio with respect to positive
    negative_ratio = [1, 2, 3, 4, 5, 6, 7, 8, None]

    kfold_ratio = pd.DataFrame()

    for i in negative_ratio:
        # Get split datasets
        X_training, y_training, X_train, y_train, X_valid, y_valid, X_test, y_test = imp_ratio(negative_ratio=i, random_state=rs, shuffle=ss)
        # do kfold
        kfold = cv_fold(all_models, X_training, y_training, X_test, y_test)
        # print(kfold) # print
        # add column for ratio
        kfold['Ratio'] = i
        kfold_ratio = pd.concat([kfold_ratio, kfold])

    # kfold_ratio.to_csv(current_dir+'/results/kfold/kfold_baseline_ratio_'+str(rs)+'_'+str(ss)+'.csv', index=False) # save the results
    return kfold_ratio


# -------------------------------------------------- Plotting
def plot(kfold_ratio, metric='Validation'):

    # List of models
    models = kfold_ratio['Name'].unique()

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line plots with markers for each model
    for model in models:
        model_data = kfold_ratio[kfold_ratio['Name'] == model]
        ratios = model_data['Ratio']

        if metric == 'Validation':
            mcc_scores = model_data['Validation MCC Score']
            mcc_std = model_data['Validation MCC Std']
        elif metric == 'Train':
            mcc_scores = model_data['Train MCC Score']
            mcc_std = model_data['Train MCC Std']
        elif metric == 'Test':
            mcc_scores = model_data['Test MCC Score']
            mcc_std = model_data['Test MCC Std']
        else:
            print('Provide a metric!')

        ax.plot(ratios, mcc_scores, marker='o', label=model)
        # ax.errorbar(ratios, mcc_scores, yerr=mcc_std, marker='o', label=model)
    
    # Set the x-axis and y-axis labels
    ax.set_xlabel('Ratio')
    ax.set_ylabel(str(metric)+' MCC Scores')

    # Set the title of the plot
    ax.set_title(str(metric)+' MCC Scores vs. Ratio')

    # Add a legend
    ax.legend()

    plt.tight_layout()

    return fig


# ------------------------------- model and plot
def kfold_ratio_model_plot(rs, ss):
    # Call the function
    kfold_ratio = kfold_ratio_model(rs, ss)
    # save the result
    kfold_ratio.to_csv(current_dir+'/results/kfold/kfold_ratio_all_'+str(rs)+'_'+str(ss)+'.csv', index=False) # save the results

    # Define the metric
    metrics = ['Train', 'Validation', 'Test']

    # Plot by calling the function
    for i in metrics:
        kfold_plot = plot(kfold_ratio, i)
        plt.show()
        kfold_plot.savefig(current_dir+'/figures/kfold/kfold_tuned_ratio_'+str(rs)+'_'+str(ss)+'_'+str(i)+'.png') # save the plot
    return None

# --------------------------------- call functions

# Parameters
rs=72 # Set random seed
ss=65 # Set shuffle seed

# kfold_ratio_model_plot(rs, ss)




# -------------------------------------------------------- combine csv

folder_path = current_dir+'/results/kfold'  # Replace with the actual folder path

# # Search for CSV files with "kfold_ratio_all" in the file name
# file_pattern = folder_path + '/kfold_ratio_all*.csv'
# csv_files = glob.glob(file_pattern)

# kfold_ratio_all_combined = pd.DataFrame()

# # Print the list of matching CSV files
# for i, file in enumerate(csv_files):
#     curr_kfold = pd.read_csv(file)
#     curr_kfold['Random'] = i
#     kfold_ratio_all_combined = pd.concat([kfold_ratio_all_combined, curr_kfold])

# kfold_ratio_all_combined.to_csv(folder_path+'/kfold_ratio_all_combined.csv', index=False)

# --------------------------------------------------------------- manipulate csv and plot

df = pd.read_csv(folder_path+'/kfold_ratio_all_combined.csv')
print(df.shape)


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

def kfold_random_state(rs, ss, note=''):
    # Get split datasets
    X_training, y_training, X_train, y_train, X_valid, y_valid, X_test, y_test = import_split_scale(random_state=rs, shuffle=ss)

    """Call the functions of the selected ML algorithms
    The parameters of each model are selected through Randomized Search CV method"""

    svc_rbf = SVC(class_weight='balanced')
    svc_lin = SVC(kernel = 'linear', class_weight='balanced')
    logistic = LogisticRegression(class_weight='balanced', max_iter=500)
    random = RandomForestClassifier(class_weight='balanced')
    knn = KNeighborsClassifier()
    xgb = XGBClassifier(scale_pos_weight=8.9038)


    # Create dictionary for each models
    svm_rbf_model = {'name':'SVM RBF','model':svc_rbf}
    svm_linear_model = {'name':'SVM Linear','model':svc_lin}
    logistic_model = {'name':'Logistic','model':logistic}
    random_model = {'name':'Random','model':random}
    knn_model = {'name':'KNN','model':knn}
    xgboost_model = {'name':'XGB','model':xgb}

    # collect all the models in a list
    all_models = [svm_rbf_model, svm_linear_model, logistic_model, random_model, knn_model, xgboost_model]

    # do kfold
    kfold = cv_fold(all_models, X_training, y_training, X_test, y_test)
    print(kfold) # print
    kfold.to_csv(current_dir+'/results/kfold/kfold_baseline_'+str(rs)+'_'+str(ss)+'_'+str(note)+'.csv', index=False) # save the results

    # Plot the results
    plot = plot_kfold_mcc(kfold) # call the function to get the plot
    plt.show() # display the plot
    plot.savefig(current_dir+'/figures/kfold/'+str(rs)+'_'+str(ss)+'_'+str(note)+'.png') # save the plot


# Set random Seed
rs = 57
ss = 65

# Class the kfold function
# kfold_random_state(rs, ss)

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

# Get combined data
combined_data = combine_kfold()

# Plot variations
plot_kfold_variation(combined_data)



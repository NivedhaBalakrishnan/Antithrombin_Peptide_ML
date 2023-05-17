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
import pickle



# Get saved models
def get_models():

    """
    Load tuned models from pickle files and return a dictionary of models.
    
    Returns:
    - models (dict): A dictionary of models where the keys are the model names and the values are the model objects.
    """
    
    # SVC RBF model
    with open(current_dir+'/dependency/pickle/untrained/svcrbf_before_fs.pkl', 'rb') as f:
        svc_rbf_model = pickle.load(f)
    svc_rbf = {'name':'svcrbf', 'model':svc_rbf_model}

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

    # KNN model
    with open(current_dir+'/dependency/pickle/untrained/knn_before_fs.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    knn = {'name':'knn', 'model':knn_model}

    # XGB model
    with open(current_dir+'/dependency/pickle/untrained/xgb_before_fs.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    xgb = {'name':'xgb', 'model':xgb_model}


    svc_rbf = {'name':'svcrbf', 'model':svc_rbf_model}
    svc_lin = {'name':'svclin', 'model':svc_lin_model}
    logistic = {'name':'logistic', 'model':logistic_model}
    random = {'name':'random', 'model':random_model}
    knn = {'name':'knn', 'model':knn_model}
    xgb = {'name':'xgb', 'model':xgb_model}

    return svc_rbf, svc_lin, logistic, random, knn, xgb



def kfold_random_state(rs, note=''):
    # Get split datasets
    X_training, y_training, X_test, y_test = import_split_scale(random_state=rs)

    """Call the functions of the selected ML algorithms
    The parameters of each model are selected through Randomized Search CV method"""

    svc_rbf_model = SVC(class_weight='balanced')
    svc_lin_model = SVC(kernel = 'linear', class_weight='balanced')
    logistic_model = LogisticRegression(class_weight='balanced', max_iter=500)
    random_model = RandomForestClassifier(class_weight='balanced')
    knn_model = KNeighborsClassifier()
    xgb_model = XGBClassifier(scale_pos_weight=8.9038)


    # Create dictionary for each models
    svc_rbf = {'name':'svcrbf','model':svc_rbf_model}
    svc_lin = {'name':'svclin','model':svc_lin_model}
    logistic = {'name':'logistic','model':logistic_model}
    random = {'name':'random','model':random_model}
    knn = {'name':'knn','model':knn_model}
    xgb = {'name':'xgb','model':xgb_model}

    all_models = [svc_rbf, svc_lin, logistic, random, knn, xgb]

    # collect all the models in a list
    # all_models = get_models()

    # do kfold
    kfold = cv_fold(all_models, X_training, y_training, X_test, y_test)
    # print(kfold) # print
    kfold.to_csv(current_dir+'/results/kfold/kfold_baseline_'+str(rs)+'_'+str(note)+'.csv', index=False) # save the results

    # # Plot the results
    plot = plot_kfold_mcc(kfold) # call the function to get the plot
    # plt.show() # display the plot
    plot.savefig(current_dir+'/figures/kfold/'+str(rs)+'_'+str(note)+'.png') # save the plot
    return


# Set random Seed
random = [42, 33, 52, 63, 77, 85, 94, 22, 65, 18]

# for i in random:
#     rs = i
#     kfold_random_state(rs, note='smote&pca')
    

# ----------------------------------------------------------------- Combine CSV

"""Note: create a new folder and save csv files corresponding to a method, and 
    give the path inside the function below
"""

def combine_kfold(note=''):
    # Create an empty DataFrame to hold the combined data
    combined_data = pd.DataFrame()

    directory = os.path.dirname(os.path.abspath(__file__))+'/results/kfold/smote&pca'
    
    print(directory)

    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

    # Loop through each CSV file, read it into a DataFrame, and concatenate it to the combined data
    for file in csv_files:
        data = pd.read_csv(file)
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Write the combined data to a new CSV file
    combined_data.to_csv(os.path.join(directory, 'combined_data'+str(note)+'.csv'), index=False)
    print(f'Number of samples: {len(combined_data)//6}')

    return combined_data

# Get combined data
# Note: Run once, since combine_data csv is placed in same repo, it add the csv for 2nd run
combined_data = combine_kfold(note='pca') # Change

# Plot variations
plot_kfold_variation(combined_data)


def get_columns_with_outliers(df):
    # Calculate Tukey's fences for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    # Convert interval column to numeric values
    df_numeric = df['SeqLength_bins'].apply(lambda x: x.mid).astype(float)
    
    # Check if each value is an outlier
    outliers_mask = (df_numeric < lower_fence) | (df_numeric > upper_fence)
    
    # Get columns with any outliers
    columns_with_outliers = df.columns[outliers_mask.any()]
    return columns_with_outliers

# Assuming you have a DataFrame called 'df'
# df = import_split_scale()
# print(df.shape)
# columns_with_outliers = get_columns_with_outliers(df)
# print(columns_with_outliers)

# Imports

# Data Analysis
import pandas as pd
import numpy as np

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import feature extraction
from data_preprocessing_packages.feature_extraction import extract_feature
from sklearn.decomposition import PCA

# save model
import os
import pickle


# Function
def import_split_scale(random_state=33):
    """
    Import the datasets, split and scale the data.

    Inputs:
    - random_state: random state for the split
    - shuffle: flag for extra shuffle

    Outputs:
    - X_training, y_training: features and target, combined set of train and validation
    - X_train, y_train: features and target of train
    - X_valid, y_valid: features and target of validation
    - X_test, y_test: features and target of test 
    """
    print(random_state)

    # The positive dataframes
    positive = pd.read_csv('data/Positive data.csv')
    negative = pd.read_csv('data/Negative data.csv')

    # extract features
    positive_data = extract_feature(positive['Seq'])
    negative_data = extract_feature(negative['Seq'])

    # print(f"Number of Samples\nPositive: {len(positive_data)}\nNegative: {len(negative_data)}\n")
    
    # Add targets
    positive_data['Class'] = 1
    negative_data['Class'] = 0

    positive_data['SeqLength_bins'] = pd.qcut(positive_data['SeqLength'], q=5)  #labels=[1,2,3,4,5]
    negative_data['SeqLength_bins'] = pd.qcut(negative_data['SeqLength'], q=5)
    
    # # Check the split
    # print(positive_data['SeqLength_bins'].value_counts())
    # print(negative_data['SeqLength_bins'].value_counts())

    # concat the dataframes
    full_data = pd.concat([positive_data, negative_data])

    # if shuffle==True:
    #     full_data = full_data.sample(frac=1, random_state=shuffle)
    
    X = full_data.drop(columns=['Class'])
    y = full_data['Class']

    # Splitting the dataset
    X_training, X_test, y_training, y_test = train_test_split(full_data, full_data['Class'], test_size=0.2, stratify=full_data[['SeqLength_bins', 'Class']], random_state=random_state)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_training, X_training['Class'], test_size=0.25, stratify=X_training[['SeqLength_bins', 'Class']], random_state=random_state)
    
    # Remove SeqLength & Class in feature set
    X_training = X_training.drop(columns=['SeqLength_bins', 'Class'])
    # X_train = X_train.drop(columns=['SeqLength_bins', 'Class'])
    # X_valid = X_valid.drop(columns=['SeqLength_bins', 'Class'])
    X_test = X_test.drop(columns=['SeqLength_bins', 'Class'])
    
    # Check counts
    # value_counts_train = y_train.value_counts()
    # value_counts_valid = y_valid.value_counts()
    # value_counts_test = y_test.value_counts()

    # print('Number of positive and negatives in each set')
    # print(f'Train\n\tPositive: {value_counts_train[1]}\n\tNegative: {value_counts_train[0]}')
    # print(f'Valid\n\tPositive: {value_counts_valid[1]}\n\tNegative: {value_counts_valid[0]}')
    # print(f'Test\n\tPositive: {value_counts_test[1]}\n\tNegative: {value_counts_test[0]}\n')

    # Scaling
    scaler = MinMaxScaler()

    # Apply Scaler to X_training
    X_training_mms = pd.DataFrame(scaler.fit_transform(X_training), index=X_training.index, columns=X_training.columns)

    # Apply same to others
    # X_train_mms = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    # X_valid_mms = pd.DataFrame(scaler.transform(X_valid), index=X_valid.index, columns=X_valid.columns)
    X_test_mms  = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # Get current directory
    current_dir = os.getcwd()
    
    # Save scaler in pickle file
    with open(current_dir+'/dependency/Scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    
    PCA
    pca = PCA()
    pca.fit(X_training_mms)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    desired_variance = 0.9
    n_components = np.argmax(cumulative_variance >= desired_variance) + 1

    # print(cumulative_variance, n_components)

    # Apply PCA with the selected number of components
    pca_selected = PCA(n_components=n_components)
    X_training_pca = pd.DataFrame(pca_selected.fit_transform(X_training_mms))
    X_test_pca = pd.DataFrame(pca_selected.transform(X_test_mms))

    # print the details
    print(f'The size of dataset\nTraining: {X_training_pca.shape}\nTest:  {X_test_pca.shape}\n')
    
    # Return 4 sets of features and target
    return X_training_pca, y_training,  X_test_pca, y_test
# X_train_mms, y_train, X_valid_mms, y_valid,
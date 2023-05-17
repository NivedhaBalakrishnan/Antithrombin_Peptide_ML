# Imports

# Data Analysis
import pandas as pd

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import feature extraction
from data_preprocessing_packages.feature_extraction import extract_feature

# save model
import os
import pickle


# Function
def import_split_scale(negative_ratio=None, random_state=33, shuffle=42, number_in_sets=False):
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

    # The positive dataframes
    positive = pd.read_csv('data/Positive data.csv')
    negative = pd.read_csv('data/Negative data.csv')

    # extract features
    positive_data = extract_feature(positive['Seq'])
    negative_data = extract_feature(negative['Seq'])

    if negative_ratio is not None:
        # Sample negative data
        sample = negative_ratio*len(positive_data)
        negative_data = negative_data.sample(n=sample, random_state=random_state)

    # total_len = len(positive_data)+len(negative_data)
    print(f"Ratio of Samples\nPositive: {len(positive_data)/len(positive_data)}\nNegative: {len(negative_data)/len(positive_data)}\n")
    
    # Add targets
    positive_data['Class'] = 1
    negative_data['Class'] = 0

    # concat the dataframes
    full_data = pd.concat([positive_data, negative_data])

    if shuffle==True:
        full_data = full_data.sample(frac=1, random_state=shuffle)
    
    X = full_data.drop(columns=['Class'])
    y = full_data['Class']

    # Splitting the dataset
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_training, y_training, test_size=0.25, stratify=y_training, random_state=random_state)
    
    value_counts_train = y_train.value_counts()
    value_counts_valid = y_valid.value_counts()
    value_counts_test = y_test.value_counts()

    if number_in_sets == True:
        # print the details
        print(f'The size of dataset\nTrain: {len(X_train)}\nValid: {len(X_valid)}\nTest:  {len(X_test)}\n')
        print('Number of positive and negatives in each set')
        print(f'Train\n\tPositive: {value_counts_train[1]}\n\tNegative: {value_counts_train[0]}')
        print(f'Valid\n\tPositive: {value_counts_valid[1]}\n\tNegative: {value_counts_valid[0]}')
        print(f'Test\n\tPositive: {value_counts_test[1]}\n\tNegative: {value_counts_test[0]}\n')

    # Scaling
    scaler = MinMaxScaler()

    # Apply Scaler to X_training
    X_training_mms = pd.DataFrame(scaler.fit_transform(X_training), index=X_training.index, columns=X_training.columns)

    # Apply same to others
    X_train_mms = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_valid_mms = pd.DataFrame(scaler.transform(X_valid), index=X_valid.index, columns=X_valid.columns)
    X_test_mms  = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # Get current directory
    current_dir = os.getcwd()
    
    # Save scaler in pickle file
    with open(current_dir+'/dependency/Scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Return 4 sets of features and target
    return X_training_mms, y_training, X_train_mms, y_train, X_valid_mms, y_valid, X_test_mms, y_test
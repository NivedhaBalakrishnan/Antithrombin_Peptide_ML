# For modules
import os

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

import pandas as pd
import matplotlib.pyplot as plt


def plot_rfe(df, name):
    plt.figure(figsize=(25,5))
    
    # Plot
    plt.scatter('Num Features', 'Train MCC Score', data=df, color='teal', s=10)
    plt.scatter('Num Features', 'Validation MCC Score', data=df, color='crimson', s=10)
    
    # Decorate
    plt.xlabel('Number of Features Selected')
    plt.ylabel('MCC Score')
    plt.legend()
    plt.title(name)
    plt.show()
    return 

# for i in ['svclin', 'logistic', 'random']:
#     rfe = pd.read_csv(current_dir+'/results/feature selection/rfe_'+i+'.csv')
#     plot_rfe(rfe, i)
def plot(df, name):
    plt.figure(figsize=(25,5))
    
    # Plot
    plt.scatter('n_estimators', 'train_means', data=df, color='teal', s=10)
    plt.scatter('n_estimators', 'test_means', data=df, color='crimson', s=10)
    
    # Decorate
    plt.xlabel('Number of Features Selected')
    plt.ylabel('MCC Score')
    plt.legend()
    plt.title(name)
    plt.show()
    return
print(current_dir)
svclin = pd.read_csv(current_dir+'/results/hypertune/random_after_fs.csv')
print(svclin.columns)

plot(svclin, name='random')


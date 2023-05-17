# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_kfold_mcc(kfold):
    fig, ax = plt.subplots(figsize=[10,5])

    ind = np.arange(len(kfold))
    width = 0.2

    train_mcc = kfold['Train MCC Score']
    val_mcc = kfold['Validation MCC Score']
    test_mcc = kfold['Test MCC Score']

    train_std = kfold['Train MCC Std']
    val_std = kfold['Validation MCC Std']
    test_std = kfold['Test MCC Std']

    rects1 = ax.bar(ind, train_mcc, width, color='teal')
    rects2 = ax.bar(ind+width, val_mcc, width, color='indianred')
    rects3 = ax.bar(ind+2*width, test_mcc, width, color='royalblue')

    ax.errorbar(ind, train_mcc, yerr=train_std, fmt='none', color='black', capsize=4)
    ax.errorbar(ind+width, val_mcc, yerr=val_std, fmt='none', color='black', capsize=4)
    ax.errorbar(ind+2*width, test_mcc, yerr=test_std, fmt='none', color='black', capsize=4)


    ax.set_ylabel('MCC', fontsize=14)
    # ax.set_yticks()
    ax.set_xticks(ind+1.5*width)
    ax.set_xticklabels(kfold['Name'], fontsize=12)
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Training', 'Validation', 'Test'))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim([30,105])

    for i, v in enumerate(train_mcc):
        ax.text(i - 0.08, v + 1, f"{v:.1f} ", fontsize=8)
        
    for i, v in enumerate(val_mcc):
        ax.text(i + width - 0.08, v + 1, f"{v:.1f}", fontsize=8)

    for i, v in enumerate(test_mcc):
        ax.text(i + 2*width - 0.08, v + 1, f"{v:.1f}", fontsize=8)
    
    # for i, v in enumerate(train_std):
    #     ax.text(i - 0.08, train_mcc[i] - 3, f"± {v:.1f}", fontsize=8, color='red')

    # for i, v in enumerate(val_std):
    #     ax.text(i + width - 0.08, val_mcc[i] - 3, f"± {v:.1f}", fontsize=8, color='red')

    # for i, v in enumerate(test_std):
    #     ax.text(i + 2*width - 0.08, test_mcc[i] - 3, f"± {v:.1f}", fontsize=8, color='red')


    plt.tight_layout()

    return fig


def plot_kfold_variation(combined_data):

    # Plot
    fig, ax = plt.subplots(1,3, figsize=[15,5])

    ax[0].scatter(x=combined_data['Name'], y=combined_data['Train MCC Score'], edgecolors='teal',alpha=0.7, color='teal')
    ax[1].scatter(x=combined_data['Name'], y=combined_data['Validation MCC Score'], edgecolors='crimson',alpha=0.7, color='crimson')
    ax[2].scatter(x=combined_data['Name'], y=combined_data['Test MCC Score'], edgecolors='royalblue',alpha=0.7, color='royalblue')
    
    ax[0].set_ylim([30,105])
    ax[1].set_ylim([30,105])
    ax[2].set_ylim([30,105])

    ax[0].tick_params(axis='x', labelsize=8)
    ax[1].tick_params(axis='x', labelsize=8)
    ax[2].tick_params(axis='x', labelsize=8)

    ax[0].set_ylabel('5-fold MCC')

    ax[0].set_title('Train')
    ax[1].set_title('Validation')
    ax[2].set_title('Test')

    plt.show()
    
    return
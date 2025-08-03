import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 


def plot_metrics(): 
    return 

def plot_cv(): 
    return 

def plot_confusion_matrix(): 
    return 

def plot_roc_curve(fpr, tpr, roc_auc):  
    fig, ax = plt.subplots() 
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    plt.show()
    return fig, ax 

def plot_precision_recall_curve(recall, precision, pr_auc):
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    plt.show()
    return fig, ax 

def plot_feature_importance(): 
    return 

def plot_correlation_matrix():
    return  
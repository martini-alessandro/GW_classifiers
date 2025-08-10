import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_cv(): 
    return 

def plot_confusion_matrix(): 
    return 

def plot_roc_curve(y_true, predicted_probabilities):
    """Plot ROC curve for the given true labels and predicted probabilities.
    Parameters:    
     - y_true: True binary labels.
    - predicted_probabilities: Predicted probabilities for the positive class."""
    #Compute ROC curve and AUC 
    fpr, tpr, thresholds = roc_curve(y_true, predicted_probabilities)
    roc_auc = auc(fpr, tpr)
    #Plot ROC Curve 
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(fpr, tpr, color='k', label='ROC curve (area = {:.4f})'.format(roc_auc))
    #Plot the diagonal line 
    ax.plot([0, 1], [0, 1], color='r', linestyle='--')
    ax.fill_between(fpr, np.repeat(0, len(fpr)), fpr, color='gray', alpha=0.2, label ='Random Guessing')
    #Set labels, title and return figure and axis
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    return fig, ax

def plot_precision_recall(y_true, preddicter_probabilities):  
    """
    Plot Precision-Recall curve for the given true labels and predicted probabilities.
    Parameters:
    - y_true: True binary labels.
    - y_pred: Predicted probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, preddicter_probabilities)
    pr_auc = auc(recall, precision)
    #Plot Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(recall, precision, color='k', label='Precision-Recall curve (area = {:.2f})'.format(pr_auc))
    #Set labels, title and return figure and axis
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    return fig, ax



def plot_feature_importance(): 
    return 

def plot_loss_function(epochs, train_loss, test_loss =None):
    fig, ax = plt.subplots() 
    
    return 

def plot_correlation_matrix():
    return  
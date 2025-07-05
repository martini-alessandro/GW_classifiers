from sklearn.metrics import precision_recall_curve, roc_curve, auc 

def auc_metric(y_true, y_pred, method = precision_recall_curve):
    """
    Compute the Area Under the Curve (AUC) for a given method.
    
    Parameters:
    - y_true: True binary labels.
    - y_pred: Predicted scores or probabilities.
    - method: Function to compute the curve (default is precision_recall_curve).
    
    Returns:
    - (precision, recoll, auc_value) if method is "precision recall"
    - (fpr, tpr, auc_value) if method is "roc"
    """
    if method == "precision recall": 
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        return precision, recall, auc(recall, precision)
    elif method == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return fpr, tpr, auc(fpr, tpr)
    else:
        raise ValueError("Unsupported method for AUC computation.")
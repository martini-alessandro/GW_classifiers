import os 
from GW_classifier.plot import * 
from GW_classifier.logger import get_logger


logger = get_logger(__name__)

def test_model(config, model, X_test, y_test): 
    """
    Tests the model on the test set and saves the results.
    
    Parameters:
    - config: Configuration dictionary containing paths and parameters.
    - model: The trained model to be tested.
    - X_test: Test features.
    - y_test: True labels for the test set.
    
    Returns:
    None
    """
    #Predict labels for test set
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else model(X_test)
    os.makedirs(os.path.join(config['savepath'], 'plots'), exist_ok=True)
    if "roc" in config['metrics']:
        logger.info("Plotting ROC curve")
        fig, ax = plot_roc_curve(y_test, y_proba[:, 1])
        fig.savefig(os.path.join(config['savepath'], 'plots', 'roc_curve.png'))
        plt.close(fig)
    if "precision recall" in config['metrics']:
        logger.info("Plotting Precision-Recall curve")
        fig, ax = plot_precision_recall(y_test, y_proba[:, 1])
        fig.savefig(os.path.join(config['savepath'], 'plots', 'precision_recall_curve.png'))
        plt.close(fig)
    if "confusion matrix" in config['metrics']:
        fig, ax = plot_confusion_matrix(y_test, y_proba[:, 1])
        fig.savefig(os.path.join(config['savepath'], 'plots', 'confusion_matrix.png'))
        plt.close(fig)
    if "feature importance" in config['metrics']:
        fig, ax = plot_feature_importance(model, X_test)
        fig.savefig(os.path.join(config['savepath'], 'plots', 'feature_importance.png'))
        plt.close(fig)
    if "loss function" in config['metrics']:
        fig, ax = plot_loss_function(model.epochs, model.train_loss, model.test_loss)
        fig.savefig(os.path.join(config['savepath'], 'plots', 'loss_function.png'))
        plt.close(fig)
    

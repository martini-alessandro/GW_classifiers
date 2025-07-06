from importlib import import_module


def load_module(path, name): 
    """
    Loads the specific module for training
    Parameters:
    -path: The path in which the module is contained (ex: sklearn.ensemble)
    -name: The name of the model to be imported (ex: RandomForestClassifier)
    Returns: 
    -The chosen model
    """
    module_path = import_module(path) 
    return getattr(module_path, name)


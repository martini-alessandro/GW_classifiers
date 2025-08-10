from importlib import import_module
import joblib 
import os 


def save_model(model, name, path = "results/models"):
    """ 
    Save a model to a file using joblib.
    Parameters:
    model (nn.Module): The PyTorch model to save.
    name (str): The name of the model file.
    path (str): The directory where the model will be saved.
    """ 
    joblib.dump(model, os.path.join(path, f"{name}.pkl"))


def load_model(name, path = "results/models"): 
    """
    Loads a model using joblib 
    Parameters: 
    -name: The name of the file 
    -path: The path where the models is stored 
    Return: 
    -The Loaded model 
    """
    return joblib.load(os.path.join(path, f"{name}.pkl"))

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
    module_name = getattr(module_path, name)
    return module_name() 

def create_directory(config): 
    """
    Creates a directory for saving results based on the configuration.
    
    Parameters:
    config (dict): Configuration dictionary containing the 'savepath'.
    
    Returns:
    str: The path to the created directory.
    """
    #Check if 'savepath' is in the config
    if not config.get('savepath'):
        raise ValueError("Configuration must contain 'savepath' key.")
    save_path = config.get('savepath') 
    #Create the save directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'pipeline'))
        os.makedirs(os.path.join(save_path, 'results'))
        os.makedirs(os.path.join(save_path, 'plots'))
    return save_path

def moveTo(obj, device): 
    """
    Moves an object to a specified device (e.g., 'cpu' or 'cuda').
    Parameters:
    obj: The object to move, which can be a tensor, list, string, dictionary, or set.
    device: The target device to move the object to.    
    Returns:
    The object moved to the specified device, or the original object if it cannot be moved.
    """
    if hasattr(obj, 'to'): 
        return obj.to(device) 
    elif isinstance(obj, list):  
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, str): 
        return (moveTo(x, device) for x in obj) 
    elif isinstance(obj, dict): 
        to_ret = {} 
        for key, value in obj.items(): 
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret 
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    else:
        return obj

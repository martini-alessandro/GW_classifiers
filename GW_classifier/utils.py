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
    module_name = getattr(module_path, name)
    return module_name() 

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

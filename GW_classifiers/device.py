import torch 

    
def moveto(obj, device): 
    """
    Moves an object to a specified device (e.g., 'cpu' or 'cuda').
    Parameters:
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
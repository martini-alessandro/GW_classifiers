from torch import nn
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



class FFNetwork(nn.module):
    """
    A simple feedforward neural network with N hidden layers.
    
    Parameters:
    input_size (int): The number of input features.
    hidden_size (int): The number of neurons in the hidden layer.
    output_size (int): The number of output classes.
    """
    
    def __init__(self, input_size, hidden_layers, output_size, activation_function=nn.ReLU, dropout = 0.0):
        super().__ini__() 

        layers = [] 
        previous_layer = input_size 

        for layer in hidden_layers:
            layers.append(nn.Linear(previous_layer, layer))
            layers.append(activation_function())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_layer = layer

        layers.append(nn.Linear(previous_layer, output_size))
        self.network = nn.Sequential(*layers) 
    
    def forward(self, x):
        return self.network(x) 
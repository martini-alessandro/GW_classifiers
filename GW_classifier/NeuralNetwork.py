from torch import nn
import os 


class FFNetwork(nn.Module):
    """
    A simple feedforward neural network with N hidden layers.
    
    Parameters:
    input_size (int): The number of input features.
    hidden_size (int): The number of neurons in the hidden layer.
    output_size (int): The number of output classes.
    """
    
    def __init__(self, input_size, hidden_layers, output_size, activation_function=nn.ReLU, dropout = 0.0):
        """
        Initializes the feedforward neural network.
        Parameters:"""
        super().__init__() 

        # Initialize the layers of the network
        layers = [] 
        previous_layer = input_size 
        for layer in hidden_layers:
            layers.append(nn.Linear(previous_layer, layer))
            layers.append(activation_function())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_layer = layer
        layers.append(nn.Linear(previous_layer, output_size))

        #Initialize the network with the layers 
        self.network = nn.Sequential(*layers) 

        # Initialize additional attributes
        self.epochs = None 
        self.training_loss = None
        self.test_loss = None
    
    def forward(self, x):
        return self.network(x) 
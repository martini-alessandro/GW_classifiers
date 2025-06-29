import torch 
from torch import nn

class FFNetwork(nn.module):
    """
    A simple feedforward neural network with N hidden layers.
    
    Parameters:
    input_size (int): The number of input features.
    hidden_size (int): The number of neurons in the hidden layer.
    output_size (int): The number of output classes.
    """
    
    def __init__(self, hidden_layers, output_size, activation_function=nn.ReLU):
        super(FFNetwork, self).__init__()
        self.module = []
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
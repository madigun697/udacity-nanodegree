import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        """Initialize a Network for DDPG.

        Params
        ======
            input_dim (int): dimension of input layer
            output_dim (int): dimension of output layer
            hidden_dims (array): array of dimensions of each hidden layer
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        assert(len(hidden_dims) >= 1)
        
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
        
        if len(self.hidden_dims) > 1:
            hidden_layers = []
            for i in range(len(hidden_dims)-1):
                hidden_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        
        output = self.output_layer(x)
        
        return output
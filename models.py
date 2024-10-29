import torch.nn as nn

class SimpleMLPTimeSeries(nn.Module):
    def __init__(self, window_size: int, variables: int = 6, hidden_dim: int = 64, output_dim: int = 12):
        super(SimpleMLPTimeSeries, self).__init__()
        
        # Calculate the flattened input size
        input_dim = window_size * variables
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Define activations
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Flatten the input from shape [batch_size, window_size, variables] to [batch_size, window_size * variables]
        x = x.view(x.size(0), -1)
        
        # Pass through the layers with activation functions
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        
        return output

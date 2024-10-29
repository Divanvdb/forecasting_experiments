import pandas as pd
import numpy as np

from io import BytesIO

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset

import requests

# Simple dataloader class

class WindSpeedDataset(Dataset):
    def __init__(self, timeseries, window_size, steps = 1, stream = 'synthetic', loader_set = 'train'):
        self.df = timeseries

        self.window_size = window_size
        self.steps = steps
        if stream is not None:
            self.data = self.df[stream].values
        else:
            self.data = self.df.values
        
        self.loader_set = loader_set

        self.split_data()
    
    def __len__(self):
        return len(self.data) - self.window_size - self.steps + 1
    
    def __getitem__(self, idx):
        return self.data[idx:idx+self.window_size], self.data[idx+self.window_size:idx+self.window_size+self.steps]
    
    def split_data(self, train_size = 0.8):
        train_size = int(len(self.data) * train_size)
        if self.loader_set == 'train':
            self.data = self.data[:train_size]
        else:
            self.data = self.data[train_size:]
    
    

# Simple MLP model

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def rollout(self, x, n):
        predictions = []
        for _ in range(n):
            pred = self.forward(x)
            
            x = torch.cat((x[:,1:], pred), dim=1)
            predictions.append(pred)

        # Predictions to numpy
        predictions = torch.cat(predictions, dim=1).detach().numpy()
        return predictions
    

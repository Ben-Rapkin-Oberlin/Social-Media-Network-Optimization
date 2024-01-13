

#import CythonMods.MH_algo as MH
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import RL_Classes as RL
import random
import torch
import torch.nn as nn
import torch.optim as optim


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        

        self.CNN = nn.Conv2d(N*N,hidden_dim,3)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.CNN = nn.Conv2d(
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


hidden_dim=256

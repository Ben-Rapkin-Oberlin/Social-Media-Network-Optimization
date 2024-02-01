
import torch
import new_graph as graph
import AC_helper as hp
import numpy as np

import torch.nn as nn

from ConvLSTM_Imp import ConvLSTMCell
from ConvLSTM_Imp import ConvLSTM

x = torch.randn(1,7,1,5,5)

#print(torch.sum(a,dim=0))
'''Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
 input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):'''
model=ConvLSTM(1,1,(2,2),3,batch_first=True)
x = torch.randn((1,7,1,5,5))
_,last_states = model(x)
print(last[0][0].shape)

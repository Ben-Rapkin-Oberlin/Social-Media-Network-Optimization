
import torch
import new_graph as graph
import AC_helper as hp
import numpy as np

import torch.nn as nn

#from ConvLSTM_Imp import ConvLSTMCell
from ConvLSTM_Imp import ConvLSTM


x=torch.randn(1,3,5,5)

y=torch.randn(1,3,5,5)
a=torch.stack((x,y),dim=1)
print(a.shape)
print(a[0].shape)
'''x=torch.randn(1,3,5,5)
hidden=torch.zeros(1,2,5,5)
combined = torch.cat([x, hidden], dim=1)
print(combined.shape)
cc_i, cc_f, cc_o, cc_g = torch.split(combined,2, dim=1)
print(cc_i.shape, cc_f.shape, cc_o.shape, cc_g.shape)
exit()
'''
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
model=ConvLSTM(6,[3,3,3,1],(3,3),4,batch_first=True)
x = torch.randn((1,7,6,5,5))
out,last_states,guess = model(x)
a=last_states[0][0]
b=last_states[0][1]
print('LL:',a.shape) #last hidden state
print('out',b.shape) #last output
c=torch.stack((a[0,0],b[0,0]),dim=0)
print(c.shape)
print(guess)
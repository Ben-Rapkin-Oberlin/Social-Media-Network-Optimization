import numpy as np
import random
import math
from RL_Classes import Environment
import matplotlib.pyplot as plt



epochs=100
N=15
K=4
seed=1
num_nodes = 100
time_steps = 8
num_epochs = 200
bits_to_flip = 1  # Number of bits to flip in each perturbation

np.random.seed(seed)
random.seed(seed)
env = Environment(num_nodes, N, K,seed)

pbounds = {}

def const_adj:

for i in range(0,num_nodes):
    for j in range(i+1,num_nodes):
        pbounds[str(i)+'_'+str(j)] = (0, 1)



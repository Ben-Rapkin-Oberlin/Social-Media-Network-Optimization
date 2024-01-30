import Original_NK as NK
import new_graph as graph

import random
import numpy as np
import torch


#TODO
#Make NK_landscape
#Make initial Graph
    #Implement a max number of connections
    #Implement a SBM clustering algorithm

#prime the simulation with R timesteps
    #Maybe just input tensor of NxNxR of 0s

#make actor crtitc
Time_Steps=10         #number of timesteps the Actor/Critic will recieve
Nodes=100             #number of nodes
Neighbors=5           #mean number of connections per node
N=15                  #number of bits in our NK landscape
K=2                   #number of bits each bit interacts with
Clusters=Nodes**(1/2) #number of clusters
Epochs=1000           #number of training epochs, may replace with episode scores
hidden_dim=(1,1,1)       #hidden dimension of ConvLSTM

landscape=NK.NKLandscape(N,K)   #make NK landscape

loops=0
#start training loop


def prime_episode(loops):

trials = 2000
psize = 100
reps = 100
n = 15
solutions = 2**n

SHARE_RATE_PARTIAL = 0.5 # Partial share
SHARE_RATE_FULL = 1.0 # Full share
SHARE_RADIUS_GLOBAL = psize - 1 # Global
SHARE_RADIUS_LOCAL = 1 # Local
NO_MUT_RATE = 0.0 # Regular, no copying mutation
MUT_RATE = 0.5    # New, half genes mutated after copying



    
    #make each ep unique
    np.random.seed(loops)
    random.seed(loops)
    torch.manual_seed(loops)

    landscape=NK.NKLandscape(N,K)
    network=graph.Population(Nodes, N, landscape, Neighbors)
    initial_genotypes = pop.genotypes.copy()
    
    pop.set_pop(initial_genotypes)
    nk.init_visited()
    pop.share_rate = exp_condition[condition][0]
    pop.share_radius = exp_condition[condition][1]
    pop.mut_rate = exp_condition[condition][2]


while True:
    #run episodes:
    """
    Each episode will initialize a new random graph, then the RL process is run until we reach some terminating state.
    Possible terminations: 
        1. ~2,500 iterations like the paper used for training
        2. Graph convergance i.e. utility is largley stable
        3. Model reaches a certaint ultility level
    """
        prim
        #make each ep unique
        np.random.seed(loops)
        random.seed(loops)
        torch.manual_seed(loops)

        landscape=NK.NKLandscape(N,K)
        network=graph.Population(Nodes, N, landscape, Neighbors)
        #run actor
        #run critic

        #update graph/get new score

        #update actor
        #update critic

        #update actor and critic inputs

        #loop  
    #check if training stopping critearia is met, conditionally loop
    loops+=1
    #for now using num of episodes, in future will look at loss rates
    if loops>=Epochs
        break






import torch
import numpy as np 
import random

import new_graph as graph


def prime_episode(loops):
    #make each ep unique
    np.random.seed(loops)
    random.seed(loops)
    torch.manual_seed(loops)

    landscape=NK.NKLandscape(N,K)
    pop=graph.Population(Nodes, N, landscape, Neighbors)

    initial_genotypes = pop.genotypes.copy()
    pop.set_pop(initial_genotypes)
    nk.init_visited()

    pop.set_community(.7,.1)
    pop.share_rate = .5 #This is what they initially define as a partial share
    pop.share_radius = 1 # I think this means only those directly connected by 1 edge
    pop.mut_rate = .5 #This is again the default

    #generate starting input
    instance=torch.zeros(1,FRAME_COUNT,1,CLUSTERS+1,CLUSTERS) #   batch, time, channel, height, width
    
    #Get avg Fitness
    inital_fit=pop.stats()

    #Make diagonal as each cluster currently has its own sudo-block
    inital_state=torch.eye(CLUSTERS,CLUSTERS)

    #Make first tensors
    instance[0,FRAME_COUNT-1,0:CLUSTERS,:]=inital_state
    instance[0,FRAME_COUNT-1,-1,:]=inital_fit
    print(instance[0,FRAME_COUNT-1,:,:])

    return pop,instance


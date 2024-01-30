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
FRAME_COUNT=10             #number of timesteps/frames the Actor/Critic will recieve
NODES=100             #number of nodes
NEIGHBORS=5           #mean number of connections per node
N=15                  #number of bits in our NK landscape
K=2                   #number of bits each bit interacts with
CLUSTERS=Nodes**(1/2) #number of clusters
EPOCHS=1000           #number of training epochs, may replace with episode scores
#hidden_dim=(1,1,1)       #hidden dimension of ConvLSTM

landscape=NK.NKLandscape(N,K)   #make NK landscape

loops=0
#start training loop

                   
    self.run(pop,nk,condition,rep,avg,meanhamm,spread,k)

    a,mh,sp = pop.stats()
    avg[condition,rep,0]=a
    meanhamm[condition,rep,0]=mh
    spread[condition,rep,0]=sp
    for trial_num in range(1,trials+1):
        pop.share(1)
        pop.learn(0)
        a,mh,sp = pop.stats()
        avg[condition,rep,trial_num]=a
        meanhamm[condition,rep,trial_num]=mh
        spread[condition,rep,trial_num]=sp


model=object #TEMP

while True:
    #run episodes:
    """
    Each episode will initialize a new random graph, then the RL process is run until we reach some terminating state.
    Possible terminations: 
        1. ~2,500 iterations like the paper used for training
        2. Graph convergance i.e. utility is largley stable
        3. Model reaches a certaint ultility level
    """
    #generates new landscape and graph, seeds with loop
    pop,instance=graph.prime_episode(loops) 


    For i in range(0,2500): #initially using paper's times steps instead of convergence

        frame=pop.step() #returns new fitness average
        #get input
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






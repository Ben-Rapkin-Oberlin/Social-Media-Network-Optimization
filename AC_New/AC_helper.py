import numpy as np 
import random

import Original_NK as NK
import new_graph as graph

import torch
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F

info=[]
#model=F.conv1d((1,1), (1,1))
#optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps= np.finfo(np.float32).eps.item()


def initialize(setup):
    global info 
    info = setup
    return

def prime_episode(loops,in_probs=.7):
    #make each ep unique
    np.random.seed(loops)
    random.seed(loops)
    torch.manual_seed(loops)

    landscape=NK.NKLandscape(info[2],info[3])

    pop=graph.Population(info[0], info[2], landscape, info[1], info[5],in_group=in_probs)

    initial_genotypes = pop.genotypes.copy()
    pop.set_pop(initial_genotypes)
    landscape.init_visited()

    pop.set_community()
    pop.share_rate = .5 #This is what they initially define as a partial share
    pop.share_radius = 1 # I think this means only those directly connected by 1 edge
    pop.mut_rate = .5 #This is again the default

    #generate starting input
    instance=torch.zeros(1,info[4],1,info[5]+1,info[5]) #   batch, time, channel, height, width
    
    #Get avg Fitness
    inital_fit, _,_=pop.stats()

    #Make diagonal as each cluster currently has its own sudo-block
    inital_state=torch.eye(info[5],info[5])

    #Make first tensors
    #print(instance[0,info[4]-1,0,0:info[5],:].shape)
    instance[0,info[4]-1,0,0:info[5],:]=inital_state
    instance[0,info[4]-1,0,-1,:]=inital_fit
    #print(instance[0,info[4]-1,0,:,:])

    return pop,instance


def update_instance(instance,act_out,avg):
    new=torch.zeros(1,info[4],1,info[5]+1,info[5]) 

    new[:,0:info[4]-1,:,:,:]=instance[:,1:,:,:,:]
    new[:,-1,:,0:-1,:]=act_out
    new[0,-1,0,-1,:]=avg

    return new






def step(action,instance,pop):

    #run pop on action
    avg,new_state=pop.step(action)

    
    instance=update_instance(instance,action,avg)

    return instance, avg



def select_action(state):
    #state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()

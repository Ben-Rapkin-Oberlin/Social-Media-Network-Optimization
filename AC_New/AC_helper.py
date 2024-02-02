import numpy as np 
import random

import Original_NK as NK
import new_graph as graph

import torch
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F


from ConvLSTM_Imp import ConvLSTM
from collections import namedtuple

info=[]
model= object
#model=F.conv1d((1,1), (1,1))
#optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps= np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
optimizer =None

def initialize(setup):
    global info 
    info = setup
    return

def make_model():
    global model, optimizer
    layers=3
    kernal=3
    hidden_channels=2
    out_dim= info[5] #NxN output
    model=ConvLSTM(1,[hidden_channels]*layers,(kernal,kernal),layers,out_dim,batch_first=True,bias=True)
    optimizer = optim.Adam(model.parameters(), lr=3e-2)
    return model

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
    new_avg=pop.step(action)

    
    instance=update_instance(instance,action,new_avg)

    return instance, new_avg


def select_action(network_state):
    #state = torch.from_numpy(state).float()
    hid, act_probs, guess = model(network_state)

    act_probs=act_probs.T #transpose for easier access
    m=Categorical(act_probs)
    chosen_actions=m.sample()


    action = torch.zeros((info[5],info[5]))
    for i,val in zip(range(0,info[5]),chosen_actions):
        action[val,i]=1

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(chosen_actions), guess))

    # the action to take (left or right)
    return action


def finish_episode():
    
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    #For now set gamma to .9 based on example 
    gamma=.9

    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, R))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]
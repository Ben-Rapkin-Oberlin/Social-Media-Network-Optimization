import torch
import numpy as np 
import random

import Original_NK as NK
import new_graph as graph


def prime_episode(loops,info):
    #make each ep unique
    np.random.seed(loops)
    random.seed(loops)
    torch.manual_seed(loops)

    landscape=NK.NKLandscape(info[2],info[3])
    pop=graph.Population(info[0], info[2], landscape, info[1])

    initial_genotypes = pop.genotypes.copy()
    pop.set_pop(initial_genotypes)
    landscape.init_visited()

    pop.set_community(.7,.1)
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


def update_instance(instance,act_out,avg,info):
    new=torch.zeros(1,info[4],1,info[5]+1,info[5]) 

    new[:,0:info[4]-1,:,:,:]=instance[:,1:,:,:,:]
    new[:,-1,:,0:-1,:]=act_out
    new[0,-1,0,-1,:]=avg

    return new





def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

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

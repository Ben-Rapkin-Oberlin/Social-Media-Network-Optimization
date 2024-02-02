import Original_NK as NK
import new_graph as graph
import AC_helper as hp
import random
import numpy as np
import torch
from collections import namedtuple

#make actor crtitc
FRAME_COUNT=10              #number of timesteps/frames the Actor/Critic will recieve
NODES=16                   #number of nodes
NEIGHBORS=5                 #mean number of connections per node
N=15                        #number of bits in our NK landscape
K=2                         #number of bits each bit interacts with
CLUSTERS=int(NODES**(1/2))  #number of clusters
EPOCHS=1000                 #number of training epochs, may replace with episode scores
#hidden_dim=(1,1,1)          #hidden dimension of ConvLSTM


#way blocks are set up we need Sqrt(N) to be a whole number

loops=0
info=[NODES,NEIGHBORS,N,K,FRAME_COUNT,CLUSTERS]
hp.initialize(info) 
ActorCritic=hp.make_model()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


running_reward = 10
for i_episode in range(20): #initially make stopping condition episode count

    # reset environment and episode reward
    pop,state=hp.prime_episode(loops) 
    ep_reward = 0

    # for each episode, only run 9999 steps so that we don't
    # infinite loop while learning
    for t in range(1, 2000):

        # select action from policy
        action = hp.select_action(state)

        # take the action
        state,reward=hp.step(action,state,pop)

        model.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    # update cumulative reward
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

    # perform backprop
    finish_episode()

    # log results
    print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i_episode, ep_reward, running_reward))

'''# check if we have "solved" the cart pole problem
                                if running_reward > :
                                    print("Solved! Running reward is now {} and "
                                          "the last episode runs to {} time steps!".format(running_reward, t))
break'''



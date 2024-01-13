import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from NN_models.RL_Classes import Environment
from NN_models.GRU import GRUNet
import matplotlib.pyplot as plt
import math

num_epochs = 200
num_nodes = 15 #need to update SBM for this to form
num_actions = 3 #number of edges added or subtracted
N=15
K=9
seed=1

env = Environment(num_nodes, N, K,seed)
model = GRUNet(num_nodes*num_nodes,256,1,2)
optimizer = optim.Adam(model.parameters())

env1=Environment(num_nodes, N, K,seed)
rewards_base,moving_avg_base=[],[]
rewards_new,moving_avg_new=[],[]
for i in range(num_epochs):
    rewards_base.append(env.step())
    if i>0:
        moving_avg_base.append(sum(rewards_base)/(i+1))



adj= torch.tensor(env.adj)
for epoch in range(num_epochs):

    

    #update graph
    env.update(actor_actions)
    
    #run timestep
    reward = env.step()
    #num_of_edges.append(env.data.edge_index.shape[1])
    rewards_new.append(reward)
    moving_avg_new.append(sum(rewards_new)/(epoch+1))
    if epoch%100==0:
        print('reward',reward)
    
    #print('critic',critic_predicted_value.shape)
    # Calculate the advantage
    #penilze reward for extreme amounts of edges, so either very large or small relative to nodes
    #reward=reward/math.log2(abs(env.data.edge_index.shape[1]-num_nodes))
        
    advantage = reward - critic_predicted_value.detach()
    #print(advantage)
    # Calculate the actor's loss (policy loss)

    actor_loss = (neg_log_probs * advantage).mean()
    aloss.append(actor_loss.item())
    #print('actor',actor_loss)
    #exit()
    # Calculate the critic's loss (value loss)
    critic_loss = (reward - critic_predicted_value).mean()#pow(2).mean()
    closs.append(critic_loss.item())
    # Combine losses
    total_loss = actor_loss + critic_loss
    tloss.append(total_loss.item())
    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
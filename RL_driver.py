import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from RL_Classes import Environment, GNNActorCritic
import matplotlib.pyplot as plt
#import torch.nn.functional as F
# Initialize the GNNActorCritic model
num_epochs = 50
num_nodes = 40 #need to update SBM for this to form
num_actions = 3 #number of edges added or subtracted
N=15
K=9
seed=1

env = Environment(num_nodes, N, K,seed)
model = GNNActorCritic(num_nodes, num_actions)
optimizer = optim.Adam(model.parameters())

env1=Environment(num_nodes, N, K,seed)
rewards_base=[]
rewards_new=[]
for i in range(num_epochs):
    rewards_base.append(env.step())



for epoch in range(num_epochs):
    data = env.data

    # Forward pass through the model
    actor_actions,neg_log_probs, critic_predicted_value = model(data)

    #update graph
    env.update(actor_actions)
    
    #run timestep
    reward = env.step()
    rewards_new.append(reward)
    if epoch%100==0:
        print('reward',reward)
    
    #print('critic',critic_predicted_value.shape)
    # Calculate the advantage
    advantage = reward - critic_predicted_value.detach()
    #print(advantage)
    # Calculate the actor's loss (policy loss)

    actor_loss = -(neg_log_probs * advantage).mean()

    # Calculate the critic's loss (value loss)
    critic_loss = (reward - critic_predicted_value).pow(2).mean()

    # Combine losses
    total_loss = actor_loss + critic_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
#graph the rewards
plt.plot(rewards_base)
plt.plot(rewards_new)
plt.show()
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from RL_Classes import Environment, GNNActorCritic

#import torch.nn.functional as F
# Initialize the GNNActorCritic model
num_epochs = 1
num_nodes = 100 #need to update SBM for this to form
num_actions = 3 #number of edges added or subtracted
N=15
K=9
gamma = 0.99  # Discount factor for future rewards
beta = 0.01   # Coefficient for the entropy term


env = Environment(num_nodes, N, K)
model = GNNActorCritic(num_nodes, num_actions)
optimizer = optim.Adam(model.parameters())

for epoch in range(num_epochs):
    data = env.data

    # Forward pass through the model
    nodes_chosen, critic_output = model(data)

    #update graph
    env.update(nodes_chosen)
    
    #run timestep
    reward=env.step()
    
    F.mse_loss(critic_output, torch.tensor([reward]))


    # Assume next_value is the value predicted for the next state
    # and done is a boolean indicating if the episode has ended
    TD_error = reward + (1 - done) * gamma * next_value - c
    

    # Calculate loss
    critic_value = critic_output.mean()  # Simplified for demonstration
    advantage = reward - critic_value
    actor_loss = -dist.log_prob(action) * advantage  # Policy gradient loss
    critic_loss = F.smooth_l1_loss(critic_value, torch.tensor([reward]))
    total_loss = actor_loss + critic_loss

    # Update model
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if done:  # Check if the episode is done
        break

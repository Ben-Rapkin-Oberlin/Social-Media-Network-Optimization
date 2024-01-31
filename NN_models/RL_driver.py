import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from RL_Classes import Environment, GNNActorCritic
import matplotlib.pyplot as plt
import math
#import torch.nn.functional as F
# Initialize the GNNActorCritic model
num_epochs = 200
num_nodes = 15 #need to update SBM for this to form
num_actions = 3 #number of edges added or subtracted
N=15
K=9
seed=1

env = Environment(num_nodes, N, K,seed)
model = GNNActorCritic(num_nodes, num_actions,seed)
optimizer = optim.Adam(model.parameters())

env1=Environment(num_nodes, N, K,seed)
rewards_base,moving_avg_base=[],[]
rewards_new,moving_avg_new=[],[]
for i in range(num_epochs):
    rewards_base.append(env.step())
    if i>0:
        moving_avg_base.append(sum(rewards_base)/(i+1))


num_of_edges = [env.data.edge_index.shape[1]]
aloss=[]
closs=[]
tloss=[]




# Run the training loop
def main():
    running_reward = 10

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


"""
for epoch in range(num_epochs):
    data = env.data

    # Forward pass through the model
    actor_actions,neg_log_probs, critic_predicted_value = model(data)

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
"""
#graph the rewards
plt.plot(aloss)
plt.plot(closs)
plt.plot(tloss)
plt.plot(rewards_new)
plt.legend(['actor','critic','total','reward'])
plt.show()

plt.plot(rewards_base)
plt.plot(rewards_new)
plt.legend(['base_raw','new_raw'])
#plt.show()

plt.plot(moving_avg_base)
plt.plot(moving_avg_new)
plt.legend(['base_avg','new_avg'])
#plt.show()


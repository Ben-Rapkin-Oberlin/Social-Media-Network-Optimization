import Original_NK as NK
import new_graph as graph
import AC_helper as hp
import random
import numpy as np
import torch


#make actor crtitc
FRAME_COUNT=10              #number of timesteps/frames the Actor/Critic will recieve
NODES=16                   #number of nodes
NEIGHBORS=5                 #mean number of connections per node
N=15                        #number of bits in our NK landscape
K=2                         #number of bits each bit interacts with
CLUSTERS=int(NODES**(1/2))  #number of clusters
EPOCHS=1000                 #number of training epochs, may replace with episode scores
#hidden_dim=(1,1,1)          #hidden dimension of ConvLSTM


loops=0
info=[NODES,NEIGHBORS,N,K,FRAME_COUNT,CLUSTERS]
#start training loop
"""
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
"""

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
    pop,instance=hp.prime_episode(loops,info) 
    
    for i in range(0,2500): #initially using paper's times steps instead of convergence

        frame=pop.step() #returns new fitness average
        #get input
        #run actor
        #run critic

        act_out,crit_out=model(instance)

        #update graph/get new score
        avg=pop.step(act_out)

        #update actor
        #update critic

        #update actor and critic inputs

        #update instance
        instance=hp.update_instance(instance,avg,act_out,info)

        #loop  

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



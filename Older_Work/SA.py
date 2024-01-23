import numpy as np
import random
import math
from RL_Classes import Environment
import matplotlib.pyplot as plt

import CythonMods.graph as graph

epochs=100
N=15
K=4
seed=1
num_nodes = 100
time_steps = 8
num_epochs = 200
bits_to_flip = 5  # Number of bits to flip in each perturbation

np.random.seed(seed)
random.seed(seed)
env = Environment(num_nodes, N, K,seed)



def perturb(matrix):
    """ Make a small random change to the matrix. """
    new_matrix = matrix.copy()
    for i in range(bits_to_flip):
        # Flip a single bit of the upper triangular matrix, not including the diagonal
        i=random.randint(0, matrix.shape[0] - 2) #final row is all 0s as diagonal hits the furthest point
        j=random.randint(i+1, matrix.shape[1] - 1)
        
        new_matrix[i, j] = 1 - new_matrix[i, j]  # Flip the boolean value
        new_matrix[j, i] = 1 - new_matrix[j, i]  # Symmetrically flip the boolean value
    return new_matrix

def simulated_annealing(matrix, initial_temp, cooling_rate, min_temp):
    scores=[]
    moving_avg_new=[]

    current_temp = initial_temp
    current_matrix = matrix
    current_score = env.step(current_matrix)
    iter=0
    while current_temp > min_temp:
        iter+=1
        new_matrix = perturb(current_matrix)
        #new_score = env.step(new_matrix)
        _,new_score=graph.step(new_matrix,env.fit_base,env.fit_score,num_nodes,N,env.land,num_nodes*2)
        
       
        print('iter',iter,'score',current_score,'temp',current_temp)
        if new_score > current_score or random.random() < math.exp((new_score - current_score) / current_temp):
            current_matrix = new_matrix
            current_score = new_score
            scores.append(current_score)
            moving_avg_new.append(sum(scores)/(len(scores)+1))
        current_temp *= cooling_rate                                                                
    

    return current_matrix, scores, moving_avg_new

# Example usage


#print(env.land.fit_table.sum()/len(env.land.fit_table))
env1=Environment(num_nodes, N, K,seed)
#rewards_base,moving_avg_base=[],[]
#for i in range(2500):
 #   rewards_base.append(env.step(env.adj))
  #  moving_avg_base.append(sum(rewards_base)/(i+1))

#print(rewards_base[-1])


optimized_matrix,rewards_new, moving_avg_new = simulated_annealing(env.adj, initial_temp=100, cooling_rate=0.975, min_temp=0.01)

env1=Environment(num_nodes, N, K,seed)
rewards_base,moving_avg_base=[],[]
for i in range(len(rewards_new)):
    rewards_base.append(env.step(env.adj))
    moving_avg_base.append(sum(rewards_base)/(i+1))


"""plt.plot(moving_avg_base)
plt.plot(rewards_base)

plt.legend(['base_avg','base_score'])
plt.show()

exit()"""
plt.plot(rewards_base)
plt.plot(rewards_new)
plt.legend(['base_raw','new_raw'])
plt.show()

plt.plot(moving_avg_base)
plt.plot(moving_avg_new)
plt.legend(['base_avg','new_avg'])
plt.show()


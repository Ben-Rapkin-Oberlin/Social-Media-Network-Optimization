
import torch
import new_graph as graph
import AC_helper as hp
import numpy as np

FRAME_COUNT=10              #number of timesteps/frames the Actor/Critic will recieve
NODES=9                   #number of nodes
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

pop,_=hp.prime_episode(1)

temp=np.copy(pop.adj_matrix)
#print(temp)

pop.step(torch.eye(info[5],info[5]))
#print('\n\n\n')
#print(aa)
print('\n')
print(np.subtract(temp,pop.adj_matrix))
#print(np.subtract(np.abs(aa),np.abs(pop.adj_matrix)))
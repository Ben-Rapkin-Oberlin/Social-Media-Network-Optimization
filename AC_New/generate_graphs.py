

import new_graph as graph
import matplotlib.pyplot as plt

import torch
import new_graph as graph
import AC_helper as hp
import numpy as np

FRAME_COUNT=10              #number of timesteps/frames the Actor/Critic will recieve
NODES=9                   #number of nodes
NEIGHBORS=5                 #mean number of connections per node
N=15                        #number of bits in our NK landscape
K=12                         #number of bits each bit interacts with
CLUSTERS=int(NODES**(1/2))  #number of clusters
EPOCHS=1000                 #number of training epochs, may replace with episode scores
#hidden_dim=(1,1,1)          #hidden dimension of ConvLSTM


#way blocks are set up we need Sqrt(N) to be a whole number

loops=0
info=[NODES,NEIGHBORS,N,K,FRAME_COUNT,CLUSTERS]
hp.initialize(info) 

pop,_=hp.prime_episode(1)

performance_old=[]
performance_new=[]
for i in range(200):
	avg,_,_=pop.step(static_edges=True)
	performance_old.append(avg)
	if i%20==0:
		print(i)

pop,_=hp.prime_episode(1)
for i in range(200):
	avg,_,_=pop.step(torch.eye(info[5],info[5]))
	performance_new.append(avg)
	#if i%10==0:
	print(i)

plt.plot(performance_old,label='old')
plt.plot(performance_new,label='new')
plt.legend(loc='upper left')
plt.show()

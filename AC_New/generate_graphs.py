

import new_graph as graph
import matplotlib.pyplot as plt
import networkx as nx
import torch
import new_graph as graph
import AC_helper as hp
import numpy as np

FRAME_COUNT=10              #number of timesteps/frames the Actor/Critic will recieve
NODES=36                    #number of nodes
NEIGHBORS=4                 #mean number of connections per node
N=15                        #number of bits in our NK landscape
K=12                         #number of bits each bit interacts with
CLUSTERS=int(NODES**(1/2))  #number of clusters
#hidden_dim=(1,1,1)          #hidden dimension of ConvLSTM


run_len=2500
num_runs=15
#way blocks are set up we need Sqrt(N) to be a whole number

loops=0
info=[NODES,NEIGHBORS,N,K,FRAME_COUNT,CLUSTERS]
hp.initialize(info) 


performance_old=[0 for x in range(run_len)]
performance_new_7=[0 for x in range(run_len)]
performance_new_8=[0 for x in range(run_len)]
#performance_new_9=[0 for x in range(run_len)]

#aa=nx.from_numpy_array(np.array(pop.adj_matrix))
#nx.draw(aa)
#plt.show
act=torch.eye(info[5],info[5])
for run in range(num_runs):
	pop,_=hp.prime_episode(run)
	for i in range(run_len):
		avg,_,_=pop.step(static_edges=True)
		performance_old[i]+=avg

	pop,_=hp.prime_episode(run,in_probs=.7)
	for i in range(run_len):
		avg,_,_=pop.step(act)
		performance_new_7[i]+=avg

	pop,_=hp.prime_episode(run,in_probs=.8)
	for i in range(run_len):
		avg,_,_=pop.step(act)
		performance_new_8[i]+=avg
	"""
	pop,_=hp.prime_episode(run,in_probs=.9)
	for i in range(run_len):
		avg,_,_=pop.step(torch.eye(info[5],info[5]))
		performance_new_9[i]+=avg"""
	print('run_number:', run)

		
performance_old=[x/num_runs for x in performance_old]
performance_new_7=[x/num_runs for x in performance_new_7]
performance_new_8=[x/num_runs for x in performance_new_8]
#performance_new_9=[x/num_runs for x in performance_new_9]

#aa=nx.from_numpy_array(np.array(pop.adj_matrix))
#nx.draw(aa)
#plt.show()
plt.plot(performance_old,label='old_'+str(num_runs)+'_'+str(run_len))
plt.plot(performance_new_7,label='new_'+str(num_runs)+'_'+str(run_len)+'_'+str(.7))
plt.plot(performance_new_8,label='new_'+str(num_runs)+'_'+str(run_len)+'_'+str(.8))
plt.plot(performance_new_9,label='new_'+str(num_runs)+'_'+str(run_len)+'_'+str(.9))
plt.legend(loc='upper left')
plt.show()

#plt.plot([n-o for n,o in zip(performance_new,performance_old)], label='diff')
#plt.legend(loc='upper left')
#plt.show()

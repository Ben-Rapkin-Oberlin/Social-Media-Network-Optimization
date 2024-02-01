

import new_graph as graph
import matplotlib.pyplot as plt
import networkx as nx
import torch
import new_graph as graph
import AC_helper as hp
import numpy as np
import pandas as pd

import seaborn as sns
import time 

FRAME_COUNT=10              #number of timesteps/frames the Actor/Critic will recieve
NODES=36                    #number of nodes
NEIGHBORS=4                 #mean number of connections per node
N=15                        #number of bits in our NK landscape
K=12                         #number of bits each bit interacts with
CLUSTERS=int(NODES**(1/2))  #number of clusters
#hidden_dim=(1,1,1)          #hidden dimension of ConvLSTM


run_len=2500
num_runs=1000
inital=0
info=[NODES,NEIGHBORS,N,K,FRAME_COUNT,CLUSTERS]
hp.initialize(info) 

all_performance_old = []
all_performance_new_7 = []
all_performance_new_8 = []

print('num_runs:', num_runs)
print('run_len:', run_len)

#aa=nx.from_numpy_array(np.array(pop.adj_matrix))
#nx.draw(aa)
#plt.show
act=torch.eye(info[5],info[5])
for run in range(inital, inital+num_runs):
	start=time.time()
	prime_time=0
	temp_performance_old = [0 for x in range(run_len)]
	temp_performance_new_7 = [0 for x in range(run_len)]
	temp_performance_new_8 = [0 for x in range(run_len)]

	a=time.time()
	pop,_=hp.prime_episode(run)
	prime_time+=time.time()-a
	for i in range(run_len):
		avg,_,_=pop.step(static_edges=True)
		temp_performance_old[i]=avg

	a=time.time()
	pop,_=hp.prime_episode(run,in_probs=.7)
	prime_time+=time.time()-a
	for i in range(run_len):
		avg,_,_=pop.step(act)
		temp_performance_new_7[i]=avg

	a=time.time()
	pop,_=hp.prime_episode(run,in_probs=.8)
	prime_time+=time.time()-a
	for i in range(run_len):
		avg,_,_=pop.step(act)
		temp_performance_new_8[i]=avg
	"""
	pop,_=hp.prime_episode(run,in_probs=.9)
	for i in range(run_len):
		avg,_,_=pop.step(torch.eye(info[5],info[5]))
		performance_new_9[i]+=avg"""
	all_performance_old.append(temp_performance_old)
	all_performance_new_7.append(temp_performance_new_7)
	all_performance_new_8.append(temp_performance_new_8)

	print('run_number:', run, 'Elapsed Time:', time.time()-start, 'prime_time:',prime_time)
	if run%15==14:
		print('saving')
		df_data = {
	    'Run': np.tile(np.arange(run_len), (run+1-inital) * 3),
	    'Performance': np.concatenate(all_performance_old + all_performance_new_7 + all_performance_new_8),
	    'Condition': ['old'] * run_len * (run+1-inital) + ['new_7'] * run_len *(run+1-inital) + ['new_8'] * run_len * (run+1-inital)
		}
		#print(df_data)
		df = pd.DataFrame(df_data)
		df.to_csv('outputs/Temp/raw_'+str(inital)+str(run_len)+'_'+str(run+1)+'.csv')



df_data = {
    'Run': np.tile(np.arange(run_len), (num_runs-inital) * 3),
    'Performance': np.concatenate(all_performance_old + all_performance_new_7 + all_performance_new_8),
    'Condition': ['old'] * run_len * (num_runs-inital) + ['new_7'] * run_len * (num_runs-inital) + ['new_8'] * run_len * (num_runs-inital)
}

df = pd.DataFrame(df_data)
df.to_csv('outputs/raw_'+str(inital)+str(run_len)+'_'+str(num_runs)+'.csv')
# Seaborn plot with confidence intervals
sns.lineplot(data=df, x='Run', y='Performance', hue='Condition', ci='sd')
plt.legend(loc='upper left')
plt.show()

'''	
			performance_old=[x/num_runs for x in performance_old]
			performance_new_7=[x/num_runs for x in performance_new_7]
			performance_new_8=[x/num_runs for x in performance_new_8]
			
			
			data = {
			    'Performance': performance_old + performance_new_7 + performance_new_8,
			    'Group': ['old'] * len(performance_old) + ['new_7'] * len(performance_new_7) + ['new_8'] * len(performance_new_8),
			    'Run': list(range(len(performance_old))) + list(range(len(performance_new_7))) + list(range(len(performance_new_8)))
			}
			
			df = pd.DataFrame(data)
			
			# Create the lineplot with confidence intervals
			sns.lineplot(data=df, x='Run', y='Performance', hue='Group')
			
			# Add legend and show plot
			plt.legend(loc='upper left')
			plt.show()
			
			
			
			
			
			'''



#performance_new_9=[x/num_runs for x in performance_new_9]
'''
#aa=nx.from_numpy_array(np.array(pop.adj_matrix))
#nx.draw(aa)
#plt.show()
plt.plot(performance_old,label='old_'+str(num_runs)+'_'+str(run_len))
plt.plot(performance_new_7,label='new_'+str(num_runs)+'_'+str(run_len)+'_'+str(.7))
plt.plot(performance_new_8,label='new_'+str(num_runs)+'_'+str(run_len)+'_'+str(.8))
#plt.plot(performance_new_9,label='new_'+str(num_runs)+'_'+str(run_len)+'_'+str(.9))
plt.legend(loc='upper left')
plt.show()

#plt.plot([n-o for n,o in zip(performance_new,performance_old)], label='diff')
#plt.legend(loc='upper left')
#plt.show()

#create lineplot
ax = sns.lineplot(x, y)
'''
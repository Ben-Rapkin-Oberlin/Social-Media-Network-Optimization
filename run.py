#!/usr/bin/env python2
import igraph as ig
import CythonMods.graph as graph
import CythonMods.direct_nk as nk
import matplotlib.pyplot as plt
import itertools
import time
import sys
import os
import numpy as np
import math
from scipy import stats
os.path.dirname(sys.executable)
# Generate Landscape
N = 15
K = 6
Nodes=100
steps=250
np.random.seed(0)
def setup(nodes,Max_Neighbourhood,N,landscape):
    seeds=np.random.normal((1+Max_Neighbourhood)/2, (1+Max_Neighbourhood)/16, 10000)
    seeds=[round(abs(x)) for x in seeds if round(abs(x))<Max_Neighbourhood and round(abs(x))>.5]
    
    randints=list(np.random.randint(1,high=nodes,size=10000))
    all=[x for x in range(0,nodes)]
    count=[0 for x in range(0,nodes)]
    edges=[]
    for i in range(0,nodes):
        num_of_edges=seeds.pop()
        for j in range(0,num_of_edges):
            a=randints.pop()
            
            if count[a]<Max_Neighbourhood and count[i]<Max_Neighbourhood and a!=i:
                if len([x for x in edges if (x[0]==a and x[1]==i) or (x[1]==a and x[0]==i)])==0:
                    count[a]+=1
                    count[i]+=1
                    edges.append([i,a])
                else:
                    j-=1
            else:
                j-=1
    #print(count)
    #print(edges)
    


    fit_base = np.random.choice([0, 1], size=(nodes,N))
    fit_score = landscape.get_fitness_array(fit_base)
    #fitness = [''.join(str(a) for a in x)for x in fitness]

    return edges,fit_base,fit_score

def prime_fitness(N,landscape):
    fit_base = np.random.choice([0, 1], size=(Nodes,Nodes))
    fit_score = landscape.get_fitness_array(fit_base)
    return np.array(fit_base),np.array(fit_score)

num_=[9,12,14]
scores=[]

for k in num_:

    land = nk.NKLandscape(N, k)
    
    g = ig.Graph.SBM(Nodes, [[.8,.05,.05,.05,.05],[.05,.8,.05,.05,.05],[.05,.05,.8,.05,.05],[.05,.05,.05,.8,.05],[.05,.05,.05,.05,.8]],
                              [20,20,20,20,20], directed=False, loops=False)
   
    adj=np.array(list(g.get_adjacency()),dtype=np.int32)

    fit_base=np.random.randint(0,2,size=(Nodes,N),dtype=np.int8)
    fit_score=np.array([land.fitness(x) for x in fit_base])
    #fit_score=nk.all_scores(fit_base,interact,land,N)

    #max_score=nk.get_globalmax(interact,land,N)
    scoresAvg=[]
    #print('max:',max_score)
    Neighbors=g.maxdegree()

    for i in range(0,steps):
        a = graph.step(adj, fit_base, fit_score, Nodes, N,land,Neighbors)
        fit_base = a
        fit_score = np.array([land.fitness(x) for x in fit_base])
        
        if i%100==0:
            print(i)
        scoresAvg.append(fit_score.sum()/Nodes)
        ########
        #here is where an RL algo would go
        #######
    scores.append(scoresAvg)#/max_score)
transformed_data, best_lambda = stats.boxcox(scores[0])
for i in scores:
    a=str(num_.pop())
    plt.plot(i,label='K='+a)
    #plt.plot(transformed_data,label=a+'boxcox')
    #plt.plot([x**(1/8) for x in i],label='K='+a)

    #plt.plot([math.exp(x) for x in i],label='log('+a+')')
plt.legend(loc='upper left')
plt.show()
plt.savefig('images/last_run.png')
exit()
#
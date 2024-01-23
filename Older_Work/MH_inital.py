import CythonMods.graph as graph
import CythonMods.direct_nk as nk
import CythonMods.MH_algo as MH
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import math.sigmoid as sigmoid
N = 15
Nodes=100
steps=250

harndess=[9,12,14]
scores=[]

for k in harndess:

    #################
    #initialize the landscape, graph, and fitness
    land = nk.NKLandscape(N, k)
    g = ig.Graph.SBM(Nodes, [[.8,.05,.05,.05,.05],[.05,.8,.05,.05,.05],[.05,.05,.8,.05,.05],[.05,.05,.05,.8,.05],[.05,.05,.05,.05,.8]],
                              [20,20,20,20,20], directed=False, loops=False)
    adj=np.array(list(g.get_adjacency()),dtype=np.int32)
    fit_base=np.random.randint(0,2,size=(Nodes,N),dtype=np.int8)
    fit_score=np.array([land.fitness(x) for x in fit_base])
    fit_score_sum_0 = fit_score.sum()
    
    #################
    #run the simulation

    for i in range(0,steps):
        #let graph evolve
        for j in range(0,3):
            fit_base=graph.blind_step(adj, fit_base, fit_score, Nodes, N,land)
            fit_score = np.array([land.fitness(x) for x in fit_base])
        #permute the graph
        fit_score_sum_1 = fit_score.sum()
        util=sigmoid(fit_score_sum_1)+sigmoid(fit_score_sum_1-fit_score_sum_0)
        adj=MH.mcmc_optimize(adj,util)

        







#transformed_data, best_lambda = stats.boxcox(scores[0])
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
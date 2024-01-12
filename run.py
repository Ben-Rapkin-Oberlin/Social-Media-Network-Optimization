#!/usr/bin/env python2
import igraph as ig
import CythonMods.graph as graph
import CythonMods.NK_landscape as nk
import matplotlib.pyplot as plt
import itertools
import time
import sys
import os
import numpy as np
import math
from scipy import stats
import CythonMods.nk_test as nk
os.path.dirname(sys.executable)
# Generate Landscape
N = 15
K = 6
Nodes=100
#Neighbors=8
steps=500
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

num_=[3,6,9,12,14]
scores=[]
#landscape = nk.NKModel(N, K, 1)
for k in num_:
    #landscape = nk.NKModel(N, k, 1)

    #edges,fit_base,fit_score=setup(Nodes,Neighbors,N,landscape)
    #g = ig.Graph(edges, directed=False)
    land = nk.generate_landscape(N, k)
    interact=nk.interaction_matrix(N,k)
    g = ig.Graph.SBM(Nodes, [[.8,.05,.05,.05,.05],[.05,.8,.05,.05,.05],[.05,.05,.8,.05,.05],[.05,.05,.05,.8,.05],[.05,.05,.05,.05,.8]],
                              [20,20,20,20,20], directed=False, loops=False)
    #layout = g.layout(layout='auto')
        # Plot the graph
    #fig, ax = plt.subplots()
    #ig.plot(g, target=ax)       
    #plot = ig.plot(g, target=ax, layout=layout)#,vertex_label=[str(x)[2:6] for x in fit_score])
    #plt.show()
    adj=np.array(list(g.get_adjacency()),dtype=np.int32)
    #print degrees of each node
    #print(np.sum(adj,axis=0))
    #adj=np.array(list(g.get_adjacency()))
    #fit_base,fit_score=prime_fitness(N,landscape)
    fit_base=np.random.randint(0,2,size=(Nodes,N),dtype=np.int8)
    fit_score=nk.all_scores(fit_base,interact,land,N)

    #print(type(fit_base))
    #print(type(fit_score))
    max_score=nk.get_globalmax(interact,land,N)
    scoresAvg=[]
    print('max:',max_score)
    Neighbors=g.maxdegree()
    #print('max:',Neighbors)
    for i in range(0,steps):
        a=graph.step(adj, fit_base, fit_score, Nodes, N,land,interact,Neighbors)
        #print(a)
        fit_base=a
        fit_score = fit_score=nk.all_scores(fit_base,interact,land,N)
        #print(fit_score)
        if i%250==0:    
            print(i)
        
        #   plot = ig.plot(g, target=ax, layout=layout,vertex_label=[str(x)[2:6] for x in fit_score])
            #print(fit_score.sum())
        scoresAvg.append(fit_score.sum()/Nodes)
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
exit()
#x=[[y,x] for x,y in zip(scoresAvg,range(0,steps))]
#print(x[0:5])
#clf = LogisticRegression(random_state=0).fit(x[0:-50],x[-50:])
#clf = LogisticRegression(random_state=0).fit([[x] for x in range(0,steps)],
#                                             scoresAvg)

#reg=clf.predict([[x] for x in range(0,steps)])
#plt.plot(reg,label="Regression")
plt.plot(scoresAvg,label="Average")
plt.legend(loc='upper left')
    #score2.append(fit_score[2])
    #score10.append(fit_score[10])
    #score25.append(fit_score[25])
    #score49.append(fit_score[49])

    #test entire update function
    #test get_fitness_array

    ####not implimented yet
    #fitness,edges,Nodes=graph.algo(fitness,edges,Nodes)
#plt.show() 
#plot fitness over time
#print(scores)
#plt.plot(score2,label="Node 2")
#plt.plot(score10,label="Node 10")
#plt.plot(score25,label="Node 25")
#plt.plot(score49,label="Node 49")
plt.plot(scoresAvg,label="Average")
#now plot with legend
plt.legend(loc='upper left')
plt.show()


exit()
g = ig.Graph(edges, directed=False)
print(fitness[:5])
print(np.count_nonzero(fitness[0]!=fitness[1]))
mat=np.array(g.get_adjacency())
#print(mat)



layout = g.layout(layout='auto')
    # Plot the graph
fig, ax = plt.subplots()
    #ig.plot(temp, target=ax)       
plot = ig.plot(g, target=ax, layout=layout)
plt.show() 
exit()





g=setup(Nodes,Neighbors)
layout = g.layout(layout='auto')
print(g.vs.degree())
    # Plot the graph
fig, ax = plt.subplots()
    #ig.plot(temp, target=ax)       
plot = ig.plot(g, target=ax, layout=layout)
plt.show() 
exit()

def a():
    landscape = nk.NKModel(N, K, 1)


    start = time.time()
    # Generate random setup
    mat, fit = graph.random_setup(Nodes,N,Neighbors)
    end = time.time()
    print(mat)
    print("Time to generate random setup: ", end - start)




    # Create a graph from the adjacency matrix
    temp = ig.Graph.Adjacency(mat.tolist(), mode=ig.ADJ_UNDIRECTED)#ig.Graph.Adjacency(mat.tolist())

    Fit_Labels=[]
    for i in range(len(fit)):
        Fit_Labels.append(str(landscape.get_fitness(fit[i]))[0:5])



    # Assign labels
    temp.vs['name'] = Fit_Labels


    # Use a layout for plotting
    layout = temp.layout('auto')

    # Plot the graph
    fig, ax = plt.subplots()
    #ig.plot(temp, target=ax)       
    plot = ig.plot(temp, target=ax, layout=layout, vertex_label=temp.vs['name'])
    plt.show() 
    return

go()
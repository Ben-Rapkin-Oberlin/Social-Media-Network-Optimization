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
os.path.dirname(sys.executable)
# Generate Landscape
N = 10
K = 7
Nodes=40
Neighbors=4
steps=40

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


landscape = nk.NKModel(N, K, 1)

edges,fit_base,fit_score=setup(Nodes,Neighbors,N,landscape)
g = ig.Graph(edges, directed=False)
layout = g.layout(layout='auto')
    # Plot the graph
fig, ax = plt.subplots()
    #ig.plot(temp, target=ax)       
plot = ig.plot(g, target=ax, layout=layout)
#plt.show() 
adj=np.array(list(g.get_adjacency()),dtype=np.int32)
#print degrees of each node
print(np.sum(adj,axis=0))
for i in range(0,steps):
    fit_base=graph.step(adj, fit_base, fit_score, Nodes, N)
    fit_score = landscape.get_fitness_array(fit_base)
    print(fit_score.sum())
    #test entire update function
    #test get_fitness_array

    ####not implimented yet
    #fitness,edges,Nodes=graph.algo(fitness,edges,Nodes)
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
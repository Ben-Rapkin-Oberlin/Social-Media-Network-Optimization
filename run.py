
#!/usr/bin/env python2
import igraph as ig
import CythonMods.graph as graph
import CythonMods.NK_landscape as nk
import matplotlib.pyplot as plt
import time
import sys
import os
os.path.dirname(sys.executable)
# Generate Landscape
N = 5
K = 2
Nodes=15
Neighbors=4

def go():   
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
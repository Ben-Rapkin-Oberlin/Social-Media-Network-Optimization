import igraph as ig

Nodes=6
#make simple graph
g = ig.Graph.SBM(Nodes, [[.6,.2,.2], [.2,.6,.2], [.2,.2,.6]],
                              [2,2,2],directed=False, loops=False)



print(g.edgelist())
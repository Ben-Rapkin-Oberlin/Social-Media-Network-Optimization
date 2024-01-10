import numpy as np
cimport numpy as cnp

#np.random.seed(seed)

cpdef random_setup(nodes, N, Neighbourhood_size):
    #makes a random graph with nodes nodes and returns the adjacency matrix
    #cdef cnp.ndarray graph = cnp.zeros((nodes,nodes))
    cdef cnp.ndarray graph = np.random.choice([0, 1], size=(nodes,nodes))
    
    graph=np.triu(graph, k=0)
    graph=graph + graph.T
    cdef int i,j
    for i in range(nodes):
        graph[i,i] = 0
        while graph[i].sum() > Neighbourhood_size:
            a=np.random.randint(0,nodes)
            graph[i,a] = 0
            graph[a,i] = 0
    
    
    #assign random fitness
    #each row is a node, each column is a element of the solution
    cdef cnp.ndarray fitness = np.random.choice([0, 1], size=(nodes,N))

    return graph, fitness



cpdef update_step(cnp.ndarray[int, ndim=2] graph, cnp.ndarray[int, ndim=2] fitness):
    cdef int i
    for i in range(graph.shape[0]):
        #find neighbours
        neighbours = cnp.where(graph[i]==1)[0]
        #find average fitness of neighbours
        average_fitness = fitness[neighbours].mean(axis=0)
        ###
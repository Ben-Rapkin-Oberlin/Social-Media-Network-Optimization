import numpy as np
cimport numpy as cnp
import itertools
import igraph as ig



cpdef step(cnp.ndarray[int, ndim=2] adj_matrix,
    cnp.ndarray[int, ndim=2]  fit_base,
    cnp.ndarray[double, ndim=1] fit_score, int nodes, N):
    #look at surrounding utility, use that to determine how much your opinion changes, randomly copy
    #opinion of one of your neighbors

    #copies min of 1 char, max of n-1 chars
    cdef double avg
    cdef int neighbor_count
    cdef int i
    cdef int j
    cdef int choosen_neighbor
    cdef cnp.ndarray rand_seed_low=np.random.randint(1,N//2,size=nodes+2)
    cdef cnp.ndarray rand_seed_high=np.random.randint(N//2,N,size=nodes+2)
    cdef cnp.ndarray rand_seed_index=np.random.randint(0,N-1,size=N*nodes)
    cdef cnp.ndarray rand_neighbor=np.random.randint(0,100,size=nodes+2)
    cdef int holder
    cdef int rand_neighbor_index = 0
    cdef int rand_index_counter = 0
    cdef cnp.ndarray neighbors=np.zeros(4, dtype=int)
    cdef double temp
    for i in range(0,nodes):
        avg=0
        neighbor_count=0
        neighbors=np.zeros(4,dtype=int)
        for j in range(0,nodes):
            if adj_matrix[i,j] == 1:
                
                avg+=fit_score[j]
                neighbor_count+=1
                neighbors[neighbor_count-1]=j

        if neighbor_count>0:
            avg=avg/neighbor_count
            #now select the neighbor to copy from:
            choosen_neighbor=rand_neighbor[rand_neighbor_index]
            rand_neighbor_index+=1
            #roulett wheel selection
            temp = 100/neighbor_count
            if choosen_neighbor<=temp:
                holder=neighbors[0]
            elif choosen_neighbor<=temp*2:
                holder=neighbors[1]
            elif choosen_neighbor<=temp*3:
                holder=neighbors[2]
            else:
                holder=neighbors[3]

            if avg>fit_score[i]:
                for i in range(rand_seed_low[i]):
                    #copy from holder to i
                    fit_base[i,rand_seed_index[rand_index_counter]]=fit_base[holder,rand_seed_index[rand_index_counter]]
                    rand_index_counter+=1
                
            else:   
                for i in range(rand_seed_high[i]):
                    #copy from holder to i
                    fit_base[i,rand_seed_index[rand_index_counter]]=fit_base[holder,rand_seed_index[rand_index_counter]]
                    rand_index_counter+=1

        
        else: #no neighbors, randomly mutate
            for i in range(rand_seed_high[i]):
                fit_base[i,rand_seed_index[rand_index_counter]]=1-fit_base[i,rand_seed_index[rand_index_counter]]
                rand_index_counter+=1
                        
    return fit_base
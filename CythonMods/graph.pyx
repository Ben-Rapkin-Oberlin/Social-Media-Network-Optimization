import numpy as np
cimport numpy as cnp
import itertools
import igraph as ig
#import CythonMods.NK_landscape as nk
import CythonMods.direct_nk as nk

cpdef step(cnp.ndarray[int, ndim=2] adj_matrix,
    cnp.ndarray[char, ndim=2]  fit_base,
    cnp.ndarray[double, ndim=1] fit_score, 
    int nodes, N,landscape,Neighbors):
    #look at surrounding utility, use that to determine how much your opinion changes, randomly copy
    #opinion of one of your neighbors

    #copies min of 1 char, max of n-1 chars
    cdef double avg
    cdef int neighbor_count
    cdef int i
    cdef int j
    cdef int choosen_neighbor
    cdef cnp.ndarray rand_seed_low=np.random.randint(4,9,size=nodes+2)
    #cdef cnp.ndarray rand_seed_low=np.random.randint(2,5,size=nodes+2)

    #cdef cnp.ndarray rand_seed_high=np.random.randint(N//2,N,size=nodes+2)
    cdef cnp.ndarray rand_seed_index=np.random.randint(0,N-1,size=N*nodes)
    cdef cnp.ndarray rand_neighbor=np.random.randint(0,Neighbors,size=nodes+2)
    cdef int holder
    cdef int rand_neighbor_index = 0
    cdef int rand_index_counter = 0
    cdef cnp.ndarray neighbors=np.zeros(Neighbors, dtype=int)
    cdef double temp
    cdef cnp.ndarray new_solution = np.zeros(N, dtype=int)
    for i in range(0,nodes):
        #print('i=',i,'fit_base=',fit_base[i])
        avg=0
        neighbor_count=0
        neighbors=np.zeros(Neighbors,dtype=int)
        new_solution=np.copy(fit_base[i])
        for j in range(0,nodes):
            try:
                if adj_matrix[i,j] == 1:
                    
                    avg+=fit_score[j]
                    neighbor_count+=1
                    neighbors[neighbor_count-1]=j
            except Exception as e:
                print("failed in neighbor count")
                print(e)
                print(i,j,nodes)
                print(len(neighbors))
                print(neighbor_count)
                print(len(fit_score))
                
                exit()

        if neighbor_count>0:
            #look at average of neighbors
            #avg=avg/neighbor_count
            #now select the neighbor to copy from:


            #choose index between 0 and neighbor_count-1
            choosen_neighbor=rand_neighbor[rand_neighbor_index]
            rand_neighbor_index+=1
            while choosen_neighbor>neighbor_count-1:
                choosen_neighbor=choosen_neighbor-neighbor_count

            #look at random neighbor
            holder=neighbors[choosen_neighbor]
            avg=fit_score[holder]
            if avg>fit_score[i]:
                # below avg 
                
                for k in range(rand_seed_low[i]):
                    #copy from holder to i for some number of chars
                    new_solution[rand_seed_index[rand_index_counter]]=fit_base[holder,rand_seed_index[rand_index_counter]]
                    rand_index_counter+=1
                
                try:
                    if landscape.fitness(new_solution)>fit_score[i]:                        
                        #print("below",landscape.get_fitness(new_solution),fit_score[i])
                        fit_base[i]=new_solution

                    else:
                        #try self learning
                        new_solution=np.copy(fit_base[i])
                        new_solution[rand_seed_index[rand_index_counter]]=1-new_solution[rand_seed_index[rand_index_counter]]
                        rand_index_counter+=1
                        if landscape.fitness(new_solution)>fit_score[i]:                                
                            #print("below_self",landscape.get_fitness(new_solution),fit_score[i]) 
                            fit_base[i]=new_solution
                        else:
                            #print("below_self_fail",landscape.get_fitness(new_solution),fit_score[i]) 
                            pass
                except:
                    print("failed")
                    print(new_solution)
                    print(i,nodes)
                    print(fit_score)
                    exit()

              
            else: 
                #if better than average, try to flip one bit
                new_solution[rand_seed_index[rand_index_counter]]=1-new_solution[rand_seed_index[rand_index_counter]]
                rand_index_counter+=1
                if landscape.fitness(new_solution)>fit_score[i]:        
                    
                    #print("above_self",landscape.get_fitness(new_solution),fit_score[i])
                    fit_base[i]=new_solution

                else:
                    #print("above_self_fail",landscape.get_fitness(new_solution),fit_score[i])
                    pass

        else: #no neighbors, randomly mutate
            for k in range(rand_seed_low[i]):
                #copy from holder to i for some number of chars
                new_solution[rand_seed_index[rand_index_counter]]=fit_base[holder,rand_seed_index[rand_index_counter]]
                rand_index_counter+=1
                if landscape.fitness(new_solution)>fit_score[i]:       
                    
                    #print("iso",landscape.get_fitness(new_solution),fit_score[i]) 
                    fit_base[i]=new_solution
            else:
                #print("iso_fail",landscape.get_fitness(new_solution),fit_score[i])
                pass
        #print('i=',i,'fit_base=',fit_base[i])
    return fit_base



cpdef blind_step(cnp.ndarray[int, ndim=2] adj_matrix,
    cnp.ndarray[char, ndim=2]  fit_base,
    cnp.ndarray[double, ndim=1] fit_score, 
    int nodes, N,landscape,Neighbors):


    cdef cnp.ndarray rand_seed_index=np.random.randint(0,N-1,size=N*nodes)
    cdef cnp.ndarray rand_neighbor=np.random.randint(0,Neighbors,size=nodes+2)
    cdef int holder
    cdef int rand_neighbor_index = 0
    cdef int rand_index_counter = 0

    solo_mutations=2
    social_mutations=4
    #all nodes copy from a random neighbor
    #in the future might use hamming distance to determine probs of which neighbor to copy from
    for i in range(0,nodes):
        #print('i=',i,'fit_base=',fit_base[i])
        neighbor_count=0
        neighbors=np.zeros(Neighbors,dtype=int)
        new_solution=np.copy(fit_base[i])
        for j in range(0,nodes):

            if adj_matrix[i,j] == 1:
                neighbor_count+=1
                neighbors[neighbor_count-1]=j
        
        if neighbor_count>0:
            #select the neighbor to copy from:
            #choose index between 0 and neighbor_count-1
            choosen_neighbor=rand_neighbor[rand_neighbor_index]
            rand_neighbor_index+=1
            while choosen_neighbor>neighbor_count-1:
                choosen_neighbor=choosen_neighbor-neighbor_count
            #look at random neighbor
            holder=neighbors[choosen_neighbor]
            for k in range(social_mutations):
                fit_base[i,rand_seed_index[rand_index_counter]]=fit_base[holder,rand_seed_index[rand_index_counter]]
                rand_index_counter+=1
        else:
            #no neighbors, randomly mutate
            #could possible mutate a char twice, but probably not a big deal
            for k in range(solo_mutations):
                fit_base[i,rand_seed_index[rand_index_counter]]=1-fit_base[i,rand_seed_index[rand_index_counter]]
                rand_index_counter+=1
    return fit_base

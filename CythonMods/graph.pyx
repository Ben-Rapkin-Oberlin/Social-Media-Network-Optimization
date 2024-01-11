import numpy as np
cimport numpy as cnp
import itertools
import igraph as ig



cpdef step(cnp.ndarray[int, ndim=2] adj_matrix,cnp.ndarray[int, ndim=2]  fit_base,cnp.ndarray[double, ndim=1] fit_score, cint nodes, N):
    #look at surrounding utility, use that to determine how much your opinion changes, randomly copy
    #opinion of one of your neighbors

    #copies min of 1 char, max of n-1 chars
    cdef int avg
    cdef int neighbor_count
    cdef int i
    cdef int choosen_neighbor
    cdef cnp.ndarray rand_seed_low=np.random.randint(1,N//2,size=nodes+2)
    cdef cnp.ndarray rand_seed_high=np.random.randint(N//2,N,size=nodes+2)
    cdef cnp.ndarray rand_seed_index=np.random.randint(0,N-1,size=N*nodes)
    cdef cnp.ndarray rand_neighbor=np.random.randint(0,100,size=nodes+2)
    cdef cnp.ndarray holder=cnp.zeros(2)
    cdef int rand_neighbor_index = 0
    cdef int rand_index_counter = 0
    cdef cnp.ndarray neighbors=np.zeros(4)
    cdef double temp
    for i in range(0,nodes)
        avg=0
        neighbor_count=0
        neighbors=cnp.zeros((1,4))
        for j in 
            if adj_matrix[i,j] == 1:
                
                avg+=fit_score[j]
                neighbor_count+=1
                neighbors[neighbor_count]=j

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
                fit_base[i,rand_seed_index[index_counter]]=fit_base[holder,rand_seed_index[index_counter]]
                index_counter+=1
            


        else:   
            for i in range(rand_seed_high[i]):
                #copy from holder to i
                fit_base[i,rand_seed_index[index_counter]]=fit_base[holder,rand_seed_index[index_counter]]
                index_counter+=1


    return fit_base


cdef roulett()



cpdef update_step(cnp.ndarray[int, ndim=2] graph, cnp.ndarray[int, ndim=2] fitness):
    cdef int i
    cdef cnp.ndarray neighbours
    for i in range(graph.shape[0]):
        #find neighbours
        neighbours = np.where(graph[i]==1)[0]
        #find average fitness of neighbours
        average_fitness = fitness[neighbours].mean(axis=0)
        

        return


class Population:
    """"""

    def __init__(self, popsize, n, landscape, community=False):
        self.popsize = popsize  # population size
        self.ng = n  # number of genes
        self.share_rate = 1.0  # recombination rate
        self.share_radius = popsize - 1  # how big your "neighborhood" is
        #Will be stochastic -- self.mut_rate = 0.0  # (E:May14) Percentage of the solution that is mutated during a copy/share
        self.landscape = landscape  # NK landscape
        self.genotypes = np.random.randint(2, size=popsize * n).reshape(popsize, n)
        self.learned = np.zeros(
            popsize, dtype=int
        )  # Keeps track of individuals who just learned
        self.shared = np.zeros(popsize, dtype=int)  # and who just shared
        self.dist_list = np.zeros(int(((popsize * popsize) - popsize) / 2))
        self.community = community



    def set_community(self, in_group, out_group):
        # sizes = np.ones(communities)*(self.popsize/communities)
        graph = nx.stochastic_block_model(
            sizes=[25, 25, 25, 25],
            p=[
                [in_group, out_group, out_group, out_group],
                [out_group, in_group, out_group, out_group],
                [out_group, out_group, in_group, out_group],
                [out_group, out_group, out_group, in_group],
            ],
        )
        self.adj_matrix = nx.to_numpy_array(graph)
        # print(np.sum(self.adj_matrix)
        # plt.clf()
        # plt.imshow(self.adj_matrix)
        # plt.show()

    def set_pop(self, genotypes):
        """Set the population genotypes to the given genotypes"""
        self.genotypes = genotypes.copy()

    def stats(self):
        """Return the average fitness of the population and the fitness of the best individual"""
        # Calculate Avg and Best Fitness
        avg = 0
        best = 0
        for i in range(self.popsize):
            fit = self.landscape.fitness(self.genotypes[i])
            self.landscape.visited(self.genotypes[i])
            avg += fit
            if fit > best:
                best = fit
        # Calculate Avg Hamming Distance
        # Also calculate Spread (i.e., number of unique solutions)
        k = 0
        unique = np.ones(self.popsize)
        for i in range(self.popsize):
            for j in range(i + 1, self.popsize):
                self.dist_list[k] = np.mean(
                    np.abs(self.genotypes[i] - self.genotypes[j])
                )
                if self.dist_list[k] == 0:
                    unique[i] = 0.0
                    unique[j] = 0.0
                k += 1
        return avg / self.popsize, np.mean(self.dist_list), np.mean(unique)

    # return a list of ints that represent an agent's neighbors
    def get_neighbors(self, ind):
        neighbors = [
            i for i in range(len(self.adj_matrix[ind])) if self.adj_matrix[ind][i] == 1
        ]
        return neighbors

    def learn(
        self, pref
    ):  # When pref=0, learning happens regardless of if sharing didn't happen
        """Update the genotypes of the population by having them learn"""
        self.learned = np.zeros(self.popsize, dtype=int)
        arr = np.arange(self.popsize)
        np.random.shuffle(arr)
        for i in arr:
            if pref == 0 or self.shared[i] == 0:
                original_fitness = self.landscape.fitness(self.genotypes[i])
                new_genotype = self.genotypes[i].copy()
                j = np.random.randint(self.ng)  # choose a random gene and "flip" it
                if new_genotype[j] == 0:
                    new_genotype[j] = 1
                else:
                    new_genotype[j] = 0
                new_fitness = self.landscape.fitness(new_genotype)
                if (
                    new_fitness > original_fitness
                ):  # If the change is better, we keep it.
                    self.learned[i] = 1
                    self.genotypes[i] = new_genotype.copy()

    def share(self, pref):
        """Update the genotypes of the population by having them share"""
        self.shared = np.zeros(self.popsize, dtype=int)
        new_genotypes = self.genotypes.copy()
        arr = np.arange(self.popsize)
        np.random.shuffle(arr)
        for i in arr:
            if pref == 0 or self.learned[i] == 0:
                if self.community:
                    j = np.random.choice(self.get_neighbors(i))
                else:
                    # Pick a neighbor in your radius
                    j = np.random.randint(
                        i - self.share_radius, i + self.share_radius + 1
                    )
                    while j == i:
                        j = np.random.randint(
                            i - self.share_radius, i + self.share_radius + 1
                        )
                    j = j % self.popsize
                # Compare fitness with neighbor
                if self.landscape.fitness(self.genotypes[j]) > self.landscape.fitness(
                    self.genotypes[i]
                ):
                    self.shared[i] = 1
                    # If neighbor is better, get some of their answers
                    for g in range(self.ng):
                        if np.random.rand() <= self.share_rate:
                            new_genotypes[i][g] = self.genotypes[j][g]
                            if np.random.rand() <= self.mut_rate:
                                new_genotypes[i][g] = np.random.randint(2)
        self.genotypes = new_genotypes.copy()

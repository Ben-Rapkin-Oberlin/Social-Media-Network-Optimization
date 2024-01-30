
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


"""
First Prime the graph with a random graph of Cluster blocks each of size Nodes/Clusters

Next take in clustering matrix and rewire edges to match clustering matrix

Then update nodes by belives of neighbors


"""
class Population:
    """"""

    def __init__(self, popsize, n, landscape, max_edge_mean, community=False):
        self.popsize = popsize  # population size
        self.ng = n  # number of genes
        self.share_rate = 1.0  # recombination rate
        self.share_radius = popsize - 1  # how big your "neighborhood" is
        self.mut_rate = 0.0  # (E:May14) Percentage of the solution that is mutated during a copy/share
        self.landscape = landscape  # NK landscape
        self.genotypes = np.random.randint(2, size=popsize * n).reshape(popsize, n)
        self.learned = np.zeros(
            popsize, dtype=int
        )  # Keeps track of individuals who just learned
        self.shared = np.zeros(popsize, dtype=int)  # and who just shared
        self.dist_list = np.zeros(int(((popsize * popsize) - popsize) / 2))
        self.community = community
        
        self.label={}
        self.node_edge_max={}

        # Generate a dict of normally dist max_neighbor_count for each node
        # May need to update scale
        samples = np.random.normal(loc=max_edge_mean, scale=1.5, size=3*popsize)

        # Round the samples to get integers
        int_samples = np.round(samples).astype(int)
        
        i=0
        for sample in int_samples:
            if sample > 0:
                self.node_edge_max[i]=sample
                i+=1           

    def clean_edge_count(self):
        #take the SBM graph and apply the dict of normally dist max_neighbor_count for each node
        for i in range(self.popsize):
            while self.adj_matrix[i].sum()>self.node_edge_max[i]:

                #randomly select an edge to remove
                ones_indices = np.where(self.adj_matrix[i] == 1)[0]
                index_to_flip = random.choice(ones_indices)

                #symmetrically remove the edge
                self.adj_matrix[i,index_to_flip]=0
                self.adj_matrix[index_to_flip,i]=0

                #in future maybe pick the ones with greatest hamming distance
                            

    def set_community(self, in_group, out_group):
        #seed maybe?
        # sizes = np.ones(communities)*(self.popsize/communities)
        block_sizes=[25, 25, 25, 25]
        graph = nx.stochastic_block_model(
            sizes=block_sizes,
            p=[
                [in_group, out_group, out_group, out_group],
                [out_group, in_group, out_group, out_group],
                [out_group, out_group, in_group, out_group],
                [out_group, out_group, out_group, in_group],
            ],
        )
        self.adj_matrix = nx.to_numpy_array(graph)

        """
        #label each node with its cluster, so we can find cluster probabilities later
        #maybe add dict of normally dist max_neighbor_count for each node, so we don't 
        #end up with all nodes having the same number of neighbors   

        """

        count=0
        cluster=0
        for i in block_sizes:
            for j in range(i):
                self.label[count]=cluster
                count+=1
            cluster+=1
        
        self.clean_edge_count()
        # print(np.sum(self.adj_matrix)
        # plt.clf()
        # plt.imshow(self.adj_matrix)
        # plt.show()
    
   


    def recluster(self, cluster_matrix):

        #we need a way to remeber the which cluster/sudoblock each node is in
        #i.e. which leader it has. 

        #check the number of possible clusters, i.e. the sudo-stochastic blocks, can change each time step.
        #For now we will define it as the var 'groups'

        num_groups=5 #TODO change this
        groups=[i for i in range(0,num_groups)]
        #we will for now let ingroup selection probability be .8 and the rest as .2/groups
        probs=[.8]+[.05]*4
        #TODO if we keep the names of the groups constant each iter, we need to shift the prob each time based on which
        #option is the 'in_group'

        for i in range(self.adj_matrix.shape[0]):
           # i is now a the node index
            node_cluster=self.label[i]

            #we will change one edge each time step


            #use roulete selection to pick a cluster
            #randomly select a node from that cluster

            #this returns the name of the group to choose
            selected_option = random.choices(groups, weights=probs, k=1)[0]
            

            
            #if that node is not already a neighbor 
                #if the node has less than max neighbors
                    #add the edge
                #if the original node is now over max neighbors
                    #randomly remove an edge
                    #in future we will pick the ones with greatest hamming distance

                #else    
                    #in the future figure out how to doing two way wethere


        # rewrire graph
        # if node has too many edges, compare hamming distance and keep the top mnc/(mnc+1) edges
        
        


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

    def learn(self, pref):  # When pref=0, learning happens regardless of if sharing didn't happen
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


class SLSim:
    def __init__(self):
        pass

    #we probably want to be doing work inside the trails loop
    def run(self,pop,nk,condition,rep,avg,meanhamm,spread,k):
        '''Run the simulation once for each trial.'''
        a,mh,sp = pop.stats()
        avg[condition,rep,0]=a
        meanhamm[condition,rep,0]=mh
        spread[condition,rep,0]=sp
        for trial_num in range(1,trials+1):
            pop.share(1)
            pop.learn(0)
            a,mh,sp = pop.stats()
            avg[condition,rep,trial_num]=a
            meanhamm[condition,rep,trial_num]=mh
            spread[condition,rep,trial_num]=sp

    def simulate(self,n,k,reps,community=False,in_group=None,out_group=None):
        '''For a single k, run the simulation many times for each condition'''
        avg = np.zeros(conditions*reps*(trials+1)).reshape(conditions,reps,trials+1)
        meanhamm = np.zeros(conditions*reps*(trials+1)).reshape(conditions,reps,trials+1)
        spread = np.zeros(conditions*reps*(trials+1)).reshape(conditions,reps,trials+1)
        for rep in range(reps):
            print("\tR: ",rep)
            nk = slnk.NKLandscape(n,k)
            if community:
                pop = slnk.Population(psize,n,nk,True)
                initial_genotypes = pop.genotypes.copy()
                for condition in range(conditions):
                    print("\t\tC: ",condition, ig[condition], og[condition])
                    pop.set_pop(initial_genotypes)
                    nk.init_visited()
                    pop.set_community(ig[condition],og[condition])
                    pop.share_rate = SHARE_RATE_PARTIAL
                    self.run(pop,nk,condition,rep,avg,meanhamm,spread,k)
            else:
                pop = slnk.Population(psize,n,nk,False)
                initial_genotypes = pop.genotypes.copy()
                for condition in range(conditions):
                    print("\t\tC: ",condition)
                    pop.set_pop(initial_genotypes)
                    nk.init_visited()
                    pop.share_rate = exp_condition[condition][0]
                    pop.share_radius = exp_condition[condition][1]
                    pop.mut_rate = exp_condition[condition][2]
                    self.run(pop,nk,condition,rep,avg,meanhamm,spread,k)
        return avg,meanhamm,spread

    def exp(self,minK,maxK,step,id,community=False):
        '''Run the experiment for all values of K and all conditions and save the data'''
        for k in range(minK,maxK+1,step):
            print("K: ",k)
            if community:
                avg,meanhamm,spread=self.simulate(n,k,reps,True)
            else:
                avg,meanhamm,spread=self.simulate(n,k,reps,False)
            np.save("avg_"+str(k)+"_"+str(id)+".npy", avg)
            np.save("meanhamm_"+str(k)+"_"+str(id)+".npy", meanhamm)
            np.save("spread_"+str(k)+"_"+str(id)+".npy", spread)

id = int(sys.argv[1])
sim = SLSim()
sim.exp(2,4,2,id)

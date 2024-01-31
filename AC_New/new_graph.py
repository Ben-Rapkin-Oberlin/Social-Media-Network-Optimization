
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import Original_NK as NK
import time
"""
First Prime the graph with a random graph of Cluster blocks each of size Nodes/Clusters

Next take in clustering matrix and rewire edges to match clustering matrix

Then update nodes by belives of neighbors


"""
class Population:
    """"""

    def __init__(self, popsize, n, landscape, max_edge_mean, clusters, community=True):
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
        self.clusters=clusters
        
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
            if i>=popsize:
                break 


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

        #we want sqrt(Nodes) communities
        prob=[]
        for i in range(self.clusters):
            a=[out_group]*i+[in_group]+[out_group]*(self.clusters-1-i)
            prob.append(a)

        block_sizes=[int(self.popsize / self.clusters) for i in range(self.clusters)]
        #print(block_sizes)
        graph = nx.stochastic_block_model(
            sizes=block_sizes,
            p=prob    
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
    
   
    def recluster(self, action):
        """go through each node, each it's real group, assign it a sudo-group according
           to the action key. Then randomly select a group to connect with. Next select
           a node to connect with. If this violates their max, then just pass. If it 
           violates your max, randomly drop one neighbor, the new node is included in 
           this selection

           TODO implement dropping via hamming distance and another protical for if a node is full of edges
        """
        IN_GROUP_PROB=.7
        OUT_GROUP_PROB=1-IN_GROUP_PROB
        mapping={}
        #loop overall all sudo-blocks to find clusters included
        #check=[[0]*self.popsize]*self.popsize
        #check=np.array(check)
        #print(check)
        for i in range(self.clusters):
            mapping[i]=[]
            for j in range(self.clusters):
                if action[i,j]==1:
                    mapping[i]=mapping[i]+[j]
        
        #remove any empty sudoblocks
        for i in list(mapping.keys()):
            if mapping[i]==[]:
                mapping.pop(i)

        #now sort nodes by their sudoclusters
        k=list(self.label.keys())
        v=list(self.label.values())
        
        sudo=[]
        for i in range(self.clusters):
            temp=[key for key,val in zip(k,v) if val in mapping[i]]
            sudo.append(temp)

        sudo_set=[set(x) for x in sudo]
        #may be biased as groups in lower numbers will always go first
        
        
        for i in range(len(sudo)):
            block_set=sudo_set[i]
            out_group_iter=[x for x in range(len(sudo)) if x!=i] #remove current block

            for current_node in block_set:

                #cache list of neighbors
                neighbor_list=self.get_neighbors(current_node)
                neighbor_set=set(neighbor_list)

                g=random.choices(['in','out'], weights=[IN_GROUP_PROB,OUT_GROUP_PROB],k=1)[0]

                if g=='in':
                    #print('inblock')
                    
                    #set diff will return elements contained in block but not neighbor_set
                    #add current node so no chance of self loop 
                    
                    neighbor_set.add(current_node)
                    #print(current_node)
                    #print(neighbor_list)
                    #print(block_set)
                    candidates=list(block_set-neighbor_set) #double cast is bad, but can't use rand to sample set
                    if candidates != []:
                        new_neighbor=random.choice(list(candidates)) 
                            
                        nnl=self.get_neighbors(new_neighbor)
                        if len(nnl)>=self.node_edge_max[new_neighbor]:
                            dropped=random.choice(nnl)
                            self.adj_matrix[new_neighbor,dropped]=0
                            self.adj_matrix[dropped,new_neighbor]=0
                            #print('3 dropped connection between ', dropped,new_neighbor)
                            #check[new_neighbor,dropped]+=1###
                            #check[dropped,new_neighbor]+=1###
                        if len(neighbor_list)>=self.node_edge_max[current_node]:
                            dropped=random.choice(neighbor_list)
                            self.adj_matrix[current_node,dropped]=0
                            self.adj_matrix[dropped,current_node]=0
                            #print('4 dropped connection between ', dropped,current_node)
                            #check[current_node,dropped]+=1###
                            #check[dropped,current_node]+=1###

                        self.adj_matrix[new_neighbor,current_node]=1
                        self.adj_matrix[current_node,new_neighbor]=1
                        #print('new connection between ', new_neighbor,current_node)
                        #check[current_node,new_neighbor]-=1###
                        #check[new_neighbor,current_node]-=1###
                    
                else:
                    #print('outblock')
                    out_block=sudo_set[random.choice(out_group_iter)]
                    candidates=list(out_block-neighbor_set)#don't need to add current as this is an out_block
                    if candidates != []:
                        new_neighbor=random.choice(candidates) #double cast is bad, but can't use rand to sample set
                        nnl=self.get_neighbors(new_neighbor)
                        if len(nnl)>=self.node_edge_max[new_neighbor]:
                            dropped=random.choice(nnl)
                            self.adj_matrix[new_neighbor,dropped]=0
                            self.adj_matrix[dropped,new_neighbor]=0
                            #print('3 dropped connection between ', dropped,new_neighbor)
                            #check[new_neighbor,dropped]+=1###
                            #check[dropped,new_neighbor]+=1###
                        if len(neighbor_list)>=self.node_edge_max[current_node]:
                            dropped=random.choice(neighbor_list)
                            self.adj_matrix[current_node,dropped]=0
                            self.adj_matrix[dropped,current_node]=0
                            #print('4 dropped connection between ', dropped,current_node)
                            #check[current_node,dropped]+=1###
                            #check[dropped,current_node]+=1###

                        self.adj_matrix[new_neighbor,current_node]=1
                        self.adj_matrix[current_node,new_neighbor]=1
                        #print('new connection between ', new_neighbor,current_node)
                        #check[current_node,new_neighbor]-=1###
                        #check[new_neighbor,current_node]-=1###
        return #check
         
        
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
                    j=-1 #original fails when a node has no neighboers, this fixes that
                    a=self.get_neighbors(i)
                    if a!=[]:
                        j = np.random.choice(a)

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
                # Compare fitness with 
                if j!=-1:
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

    def step(self,action=None,static_edges=False):
        #update connections based on model actions
        #action is a CLUSTERxCLUSTER matrix where each column represents a block/leader and each row represents the
        #sudo-block they've been placed into
        
        if action==None and not static_edges:
            print('input error')
            exit()
        if not static_edges:
            aa=self.recluster(action)


        #run simulation

        self.share(1)
        self.learn(0)
        a,mh,sp=self.stats()
        return a,mh,sp
            



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


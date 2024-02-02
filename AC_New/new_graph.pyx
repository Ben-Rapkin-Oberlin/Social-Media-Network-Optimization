
cimport numpy as np
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

    def __init__(self, popsize, n, landscape, max_edge_mean, clusters, community=True,in_group=.7):
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
        
        self.IN_GROUP_PROB=in_group
        self.OUT_GROUP_PROB=(1-in_group)/(self.clusters-1)
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
                            

    def set_community(self):
        #seed maybe?
        # sizes = np.ones(communities)*(self.popsize/communities)
        in_group=self.IN_GROUP_PROB
        out_group=self.OUT_GROUP_PROB

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
        #at this point, mapping takes in a sudoblock and tells you which leaders/clusters are in them


        #remove any empty sudoblocks
        for i in list(mapping.keys()):
            if mapping[i]==[]:
                mapping.pop(i)

        #now sort nodes by their sudoclusters
        k=list(self.label.keys())
        v=list(self.label.values())

        #print(action)

        a=mapping.keys()
        #print([(x,mapping[x]) for x in a])
        #print('key',k)
        #print('val',v)

        
        #now that we know which clusters are in each sudo-block,
        #make a list of nodes for easy access
        sudo=[]
        for i in mapping.keys():
            temp=[key for key,val in zip(k,v) if val in mapping[i]]
            sudo.append(temp)

        #print('sudo',sudo)
        #exit()
        sudo_set=[set(x) for x in sudo]
        #may be biased as groups in lower numbers will always go first
        
        
        for i in range(len(sudo)):
            block_set=sudo_set[i]
            out_group_iter=[x for x in range(len(sudo)) if x!=i] #remove current block

            for current_node in block_set:

                #cache list of neighbors
                neighbor_list=self.get_neighbors(current_node)
                neighbor_set=set(neighbor_list)

                #make sure there is an outgroup, model may put everyone together
                if out_group_iter!=[]:
                    g=random.choices(['in','out'], weights=[self.IN_GROUP_PROB,1-self.IN_GROUP_PROB],k=1)[0]
                else:
                    g='in'

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
        avg,mh,sp=self.stats()
        return avg
            

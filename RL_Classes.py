#using Actor-Critic method

#import CythonMods.graph as graph
import CythonMods.direct_nk as nk
#import CythonMods.MH_algo as MH
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
#import math.sigmoid as sigmoid
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import aggr
class Environment:
    def __init__(self,Nodes, N, K,seed):
        #base parameters
        self.N = N
        self.K = K
        self.Nodes = Nodes
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        #initialize the landscape, graph, and fitness
        self.land = nk.NKLandscape(self.N, self.K)
        self.g = ig.Graph.SBM(self.Nodes, [[.8,.05,.05,.05,.05],[.05,.8,.05,.05,.05],[.05,.05,.8,.05,.05],[.05,.05,.05,.8,.05],[.05,.05,.05,.05,.8]],
                                [20,20,20,20,20], directed=False, loops=False)
        #do above for 40 nodes
        #self.g = ig.Graph.SBM(self.Nodes, [[.85,.05,.05,.05],[.05,.85,.05,.05],[.05,.05,.85,.05],[.05,.05,.05,.85]],
         #                       [10,10,10,10], directed=False, loops=False)
        #self.g = ig.Graph.SBM(self.Nodes, [[.7,.1,.1],[.1,.7,.1],[.1,.1,.7]],
        #                        [5,5,5], directed=False, loops=False)
        #generate a random graph of Nodes nodes
        #self.g = ig.Graph.Erdos_Renyi(n=self.Nodes, m=2*self.Nodes, directed=False, loops=False)
        #get hightest degree node
        #print(list(self.g.degree()).sort())
        self.adj=np.array(list(self.g.get_adjacency()),dtype=np.int32)
        self.fit_base=np.random.randint(0,2,size=(Nodes,N),dtype=np.int8)
        self.fit_score=np.array([self.land.fitness(x) for x in self.fit_base])
        self.fit_score_sum_0 = self.fit_score.sum()

        self.edge_list = torch.tensor([edge.tuple for edge in self.g.es], dtype=torch.long).t().contiguous()
        self.edge_map ={} # so we do not need to search for the edge index to remove the edge
        for i in range (self.edge_list.shape[1]):
            self.edge_map[self.edge_list[0,i].item(),self.edge_list[1,i].item()]=i
        
        #print('map',len(self.edge_map.keys()))
        #print('adj',int(sum([x.sum() for x in self.adj])/2))

        #print(self.edge_list)
        # Node features - using identity features here
        num_nodes = Nodes #len(self.g.vs)
        node_features = torch.eye(num_nodes)
        #print
        # Create torch_geometric data
        self.data = Data(x=node_features, edge_index=self.edge_list)

    def update(self, actions):
        #update the adj and graph object
        #if self.data.edge_index.shape[1]>=self.Nodes*5:
         #   return
        for edge in actions:         
            #print('map',len(self.edge_map.keys()))
            #print('adj',int(sum([x.sum() for x in self.adj])/2))
            #flip bit
            i,j=edge[0].item(),edge[1].item()
            #print('isum:',self.adj[i].sum())
            a=1-self.adj[i,j]
            #print(a)
            self.adj[i,j]=a
            self.adj[j,i]=a
            #print('i:',i,'j:',j)
            #update graph object
            if a==1:
                #add edge
                new_edge = torch.tensor([[i,j], [j,i]], dtype=torch.long)
                self.data.edge_index = torch.cat([self.data.edge_index, new_edge], dim=1)
                self.edge_map[i,j]=self.data.edge_index.shape[1]-2
            else:
                #remove edge
                #print('update_edge_index:', self.data.edge_index.shape)
                temp=[(x.item(),y.item()) for x,y in zip(self.data.edge_index[0,:],self.data.edge_index[1,:])]
                #print('num of entries',[x for x in temp if x[0]==i or x[1]==i])
                #print(self.edge_map[(i,j)])
                try:
                    self.data.edge_index = torch.cat([self.data.edge_index[:,:self.edge_map[i,j]], self.data.edge_index[:,self.edge_map[i,j]+1:]], dim=1)
                    del self.edge_map[i,j]
                except:
                    self.data.edge_index = torch.cat([self.data.edge_index[:,:self.edge_map[j,i]], self.data.edge_index[:,self.edge_map[j,i]+1:]], dim=1)
                    del self.edge_map[j,i]
                    #print('except succ')
        return 


    def reset(self):
        #reset the graph, and fitness, but not the landscape
        self.g = ig.Graph.SBM(self.Nodes, [[.8,.05,.05,.05,.05],[.05,.8,.05,.05,.05],[.05,.05,.8,.05,.05],[.05,.05,.05,.8,.05],[.05,.05,.05,.05,.8]],
                                [20,20,20,20,20], directed=False, loops=False)
        self.adj=np.array(list(self.g.get_adjacency()),dtype=np.int32)
        self.fit_base=np.random.randint(0,2,size=(Nodes,N),dtype=np.int8)
        self.fit_score=np.array([self.land.fitness(x) for x in self.fit_base])
        self.fit_score_sum_0 = self.fit_score.sum()

        return self.matrix

    def step(self,mat):
        #blind step, no knowledge of own/surrounding fitness
        Neighbors = self.Nodes #max num of edges a node may have
        rand_seed_index=np.random.randint(0,self.N-1,size=self.N*self.Nodes)
        rand_neighbor=np.random.randint(0,20,size=self.Nodes+2)
        holder=0
        rand_neighbor_index = 0
        rand_index_counter = 0

        solo_mutations=1
        social_mutations=4
        #all nodes copy from a random neighbor
        #in the future might use hamming distance to determine probs of which neighbor to copy from
        for i in range(0,self.Nodes):
            #print('i=',i,'fit_base=',fit_base[i])
            neighbor_count=0
            neighbors=np.zeros(Neighbors,dtype=int)
            new_solution=np.copy(self.fit_base[i])
            for j in range(0,self.Nodes):

                #if self.adj[i,j] == 1:
                if mat[i,j] == 1:
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
                    self.fit_base[i,rand_seed_index[rand_index_counter]]=self.fit_base[holder,rand_seed_index[rand_index_counter]]
                    rand_index_counter+=1
            else:
                #no neighbors, randomly mutate
                #could possible mutate a char twice, but probably not a big deal
                for k in range(solo_mutations):
                    self.fit_base[i,rand_seed_index[rand_index_counter]]=1-self.fit_base[i,rand_seed_index[rand_index_counter]]
                    rand_index_counter+=1
        self.fit_score=np.array([self.land.fitness(x) for x in self.fit_base])
        #total_score = self.fit_score.sum()#
        total_score = self.fit_score.mean()
        return total_score



    def prime(self,time_steps):
        #prime the graph with a few steps
        starter=[]
        for i in range(time_steps):
            tempA=self.adj
            score=self.step() #this is the score
            tempB = np.full((1, self.Nodes), score)  # Create a 1xN array filled with 'score'
        
            # Stack tempB below tempA
            tempA = np.vstack((tempA, tempB))
            
            #tempA=np.append(tempA,tempB,axis=1)
            starter.append(tempA)

        return np.array(starter)
    
    def smart_step(mat):
        pass
        




####################################################################
# GNN Actor-Critic                                                 #
#                                                                  #
####################################################################

class GNNActorCritic(nn.Module):
    def __init__(self, N,X,seed):
        super(GNNActorCritic, self).__init__()
        self.N = N
        self.X = X
        torch.manual_seed(seed)
        #Not sure if next is correct
        num_node_features=N
        # Common GNN layers
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 20)
        

        # Actor layers
        self.actor_fc = nn.Linear(20, N*N)

        # Critic layers
        self.critic_fc = nn.Linear(20, 1)
        #self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)

        self.saved_actions = []
        self.rewards = []

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        #print(x.shape)  
        #print(edge_index.shape)
        # Common GNN layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        #print(x.shape)

        # Critic
        critic_output = self.critic_fc(x)
        critic_output = critic_output.mean(dim=0)
        #print('co',critic_output.shape)
        # Actor
        # Global pooling (e.g., mean pooling)
        x_actor = torch.mean(x, dim=0)
        # Apply fully connected layer to get probabilities over node pairs
        logits = self.actor_fc(x_actor) 
        logits = logits.view(self.N, self.N)

        # Apply softmax to get probabilities
        maxx,mi,mj=0,0,0
        smax,si,sj=0,0,0
        probabilities = F.softmax(logits, dim=1)
        for i in range(0,self.N):
            for j in range (0,self.N):
                #if i==j:
                    #probabilities[i,j]=0
                if probabilities[i,j]>maxx:
                    smax=maxx
                    si, sj = mi, mj
                    mi, mj = i, j
                    maxx=probabilities[i,j]
        actions=torch.tensor([[mi,mj],[si,sj]])                    

        # Calculate log probabilities of the chosen actions
        log_probs = -1*torch.log(probabilities.view(-1)[actions])

        #print('actions:',actions.shape)
        #print('log_probs:',log_probs.shape)
        
        return actions,log_probs, critic_output
    

import Original_NK as NK

#TODO
#Make NK_landscape
#Make initial Graph
    #Implement a max number of connections
    #Implement a SBM clustering algorithm

#prime the simulation with R timesteps
    #Maybe just input tensor of NxNxR of 0s

#make actor crtitc
Time_Steps=10         #number of timesteps the Actor/Critic will recieve
Nodes=100             #number of nodes
Neighbors=5           #max number of connections per node
N=15                  #number of bits in our NK landscape
K=2                   #number of bits each bit interacts with
Clusters=Nodes**(1/2) #number of clusters
Epochs=1000           #number of training epochs, may replace with episode scores
hidden_dim=(1,1,1)       #hidden dimension of ConvLSTM

landscape=NK.NKLandscape(N,K)   #make NK landscape


#start training loop
    #run episodes:
    """
    Each episode will initialize a new random graph, then the RL process is run until we reach some terminating state.
    Possible terminations: 
        1. ~2,500 iterations like the paper used for training
        2. Graph convergance i.e. utility is largley stable
        3. Model reaches a certaint ultility level
    """
        #run actor
        #run critic

        #update graph/get new score

        #update actor
        #update critic

        #update actor and critic inputs

        #loop  
    #check if training stopping critearia is met, conditionally loop
    



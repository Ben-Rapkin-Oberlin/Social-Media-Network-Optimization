import torch.nn as nn
import torch

from ConvLSTM_Imp import ConvLSTM


#Current Plan:
#Actor is a ConvLSTM which outputs 
#The critic will be a CNN which accepts the output of Actor 
#and a tensor of the last X scores
#The critic will output a single value


#make CNN for critic
#Current dimensions are temporary
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3,stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    


def make_actorCritc(channels):
    actor = ConvLSTM(input_dim=channels,
                 #hidden_dim=[64, 64, 128],
                 #I am not yet sure what we want hidden dimensions to be
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True
                 bias=True,
                 return_all_layers=False)
    

    #takes in list of last X scores and new suggested action and outputs a single value
    critic = Critic(input_dim=#not sure specifically yet)
    )

    return actor, critic

    
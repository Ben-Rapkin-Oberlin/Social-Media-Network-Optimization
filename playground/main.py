
import torch
import torch.nn as nn
from torch.distributions import Categorical
N=2
probabilities = torch.softmax(torch.rand(N, N), dim=0)
probabilities = torch.Tensor([[.01,.01,.01],[.01,.02,.98],[.98,.97,.01]]).T

# Sample one action for each agent
# torch.multinomial expects weights in a 2D tensor where each row contains the weights and
# we want to sample from columns, so we transpose the tensor.
# No replacement since we're sampling one action per agent.
print(probabilities)
m=Categorical(probabilities)
a=m.sample()
print(a)
print(m.log_prob(a))

'''torch.log_prob
print(probabilities)
actions = torch.multinomial(probabilities.t(), 1).squeeze()
logp= torch.log(probabilities)
print(actions)
print(logp)
aa = torch.zeros((N,N))
for i,val in zip(range(0,N),actions):
	aa[val,i]=1

print(aa)'''
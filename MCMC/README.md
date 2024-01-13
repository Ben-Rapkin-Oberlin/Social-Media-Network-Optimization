
At the moment I am confused how we are defining the utility function if nodes do not copy one another each step as our global health is dependent on the node values which the MCMC cannot change.

 Perhaps he is suggesting MCMC for NK-solving, but that is not at all novel. So I am going to do a simple implementation which runs on dumb nodes which cannot see their own utility

 Objective Funct= F(avg Utility)+H(change in ave utility)

At the moment I do not think basic MH/Gibbs/SA will be great as in theory the objective function might not be stable due to the delta function. So I will try to first implement Particle Filters/SMC


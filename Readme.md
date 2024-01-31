### Overview  

Currently work is being done on implementing an actor critic model with the clustering action space. Recluster needs testing on non-diagonal matrices, the training loop needs some edits, and the models need tailoring. Currently on track. Development can be found in the AC_New (Actor_Critic) directory.  


To run 

``` 

$ cd AC_New 

$ python AC_driver.py 

``` 
### Important Files 

```bash
AC_New
├── AC_driver.py:         File outlines the training loop and calls high level functions
├── AC_helper.py:         File containing technical elements of the training loop
├── Autoencoder.py:       File containing the actor critic class
├── ConvLSTM_Imp.py:      File containing the classes implementing a ConvLSTM in pytorch
├── Original_NK.py:       File containing the original nk_landscape code
├── generate_graphs.py:   File which generates diagrams
├── new_graph.py:         File containing the updated version of the partial copying social network
├── outputs:              Directory containing performance graphs
└── tester.py:            File containing utility/testing script
```

### Folder Outlines 
**Images** Contains graphs of first/old AC trials 

**MCMC** For future implementations of Markov Decision Processes 

**Older_Work** attempt 1 at the AC model and an adapted GRU model   

**CythonMods** Cythoized versions of basic utility functions used in the first AC model and Simulated annealing 
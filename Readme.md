### Overview  

Currently work is being done on implementing an actor critic model with the clustering action space. Recluster needs testing on non-diagonal matrices, the training loop needs some edits, and the models need tailoring. Currently on track. Development can be found in the AC_New (Actor_Critic) directory.  


To run 

``` 

$ cd AC_New 

$ python AC_driver.py 

``` 
### Important Files 

```bash
├── AC_New
│   ├── AC_driver.py
│   ├── AC_helper.py
│   ├── Autoencoder_model.py
│   ├── ConvLSTM_Imp.py
│   ├── Original_NK.py
│   ├── __pycache__
│   │   ├── AC_helper.cpython-310.pyc
│   │   ├── Original_NK.cpython-310.pyc
│   │   └── new_graph.cpython-310.pyc
│   ├── generate_graphs.py
│   ├── new_graph.py
│   ├── original_graph
│   │   ├── SLNK.py
│   │   └── Sim_SL.py
│   ├── outputs
│   │   └── 10_1000_.7.png
│   └── tester.py

```
**AC_Driver** outlines the training loop and calls high level functions 

**AC_helper** does most of the more technical elements of the training loop and interfaces with the other files   

**new_graph** Contains the updated version of the partial copying social network 

**Original_NK** Contains the original nk_landscape code 

**ConvLSTM** Contains the classes implementing a ConvLSTM in pytorch 

**Autoencoder** Contain the actor critic class 

**tester** is a utility script and area to run functions 


### Folder Outlines 
**Images** Contains graphs of first AC trials 

**MCMC** For future implementations of Markov Decision Processes 

**Older_Work** attempt 1 at the AC model and an adapted GRU model   

**CythonMods** Cythoized versions of basic utility functions used in the first AC model and Simulated annealing 
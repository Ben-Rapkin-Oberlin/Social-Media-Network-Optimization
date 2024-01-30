


### Overview 
Currently work is being done on implementing an actor critic model with the clustering action space. Recluster needs testing on non-diagonal matrices, the training loop needs some edits, and the models need taloring. Currently on track

Current Development Can befound in the AC_New (Actor_Critic) directory. 

### Files
*AC_Driver* outlines the training loop and calls high level functions

*AC_helper* does most of the more technical elements of the training loop and interfaces with the other files

*new_graph* Contains the updated version of the partial copying social network

*Original_NK* Contains the original nk_landscape code

*ConvLSTM* Contains the classes implementing a ConvLSTM in pytorch

*Autoencoder* Contain the actor critic class

*tester* is a utility script and area to test functions
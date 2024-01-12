import numpy as np
import CythonMods.nk_test as nk


np.random.seed(0)
land = nk.generate_landscape(5, 2)
interact=nk.interaction_matrix(5,2)
vec=np.random.randint(0,2,size=(3,5),dtype=np.int8)
#a=nk.landscapes.generate_landscape(1,5,2,0,0,0)
print(land)
print(vec)
scores=nk.all_scores(vec,interact,land,5)
max=nk.get_globalmax(interact,land,5)
print(scores)
print(max)

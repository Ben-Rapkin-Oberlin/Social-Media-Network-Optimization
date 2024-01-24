    

import numpy as np
import random


arr=np.array([1,0,1,0])

ones_indices = np.where(arr == 1)[0]
index_to_flip = random.choice(ones_indices)
print(ones_indices)
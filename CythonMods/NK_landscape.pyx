
import numpy as np
cimport numpy as cnp

cdef class NKModel:
    cdef public int N, K
    cdef public cnp.ndarray landscapes

    def __init__(self, int N, int K, int seed=1):
        self.N = N
        self.K = K
        #if seed is not None:
            #np.random.seed(seed)
        self.landscapes = self.generate_landscapes(N, K)

    cdef cnp.ndarray generate_landscapes(self, int N, int K):
        # Each row in the landscape array represents one gene's local fitness landscape
        cdef cnp.ndarray landscapes = np.random.uniform(0, 1, (N, int(2**(K+1))) )#.astype(np.float64)
        return landscapes


    cpdef double get_fitness(self, cnp.ndarray[int, ndim=1] state):
        cdef int i, idx
        cdef double total_fitness = 0

        for i in range(self.N):
            # Create the index for the row in the landscape array
            idx = state[i] * int(2**self.K)
            for j in range(self.K):
                idx += state[(i+j+1) % self.N] * 2**(self.K-j-1)

            # Add the local fitness to the total fitness
            total_fitness += self.landscapes[i, idx]

        return (total_fitness / self.N)**8
    
    cpdef double get_fitness_array(self, cnp.ndarray[int, ndim=2] states):
        cdef int i, j, idx
        cdef double total_fitness = 0
        cdef int num_states = states.shape[0]
        cdef cnp.ndarray fitnesses = np.zeros(num_states)

        for i in range(num_states):
            total_fitness = 0
            for j in range(self.N):
                # Create the index for the row in the landscape array
                idx = states[i, j] * int(2**self.K)
                for k in range(self.K):
                    idx += states[i, (j+k+1) % self.N] * 2**(self.K-k-1)

                # Add the local fitness to the total fitness
                total_fitness += self.landscapes[j, idx]

            fitnesses[i] = (total_fitness / self.N)**8

        return fitnesses

# Example usage
def create_nk_landscape(int N, int K):
    cdef NKModel nk_model = NKModel(N, K)
    return nk_model
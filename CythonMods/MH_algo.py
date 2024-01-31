import numpy as np

update_num = 8
Nodes=100
step_num=2000
index_seed=np.random.randint(0, Nodes, size=step_num*update_num+10)
index_seed_counter=0

def util(matrix):
    pass

def propose_new_matrix(matrix):
    # Propose a new matrix by flipping a random element
    # not sure if this counts as MH
    new_matrix = np.copy(matrix)
    for i in range(update_num):
        i,j = index_seed[index_seed_counter], index_seed[index_seed_counter+1]
        index_seed_counter+=2
        new_matrix[i,j] = 1-matrix[i,j]
        new_matrix[j,i] = 1-matrix[j,i]
    return new_matrix


def acceptance_probability(old_utility, new_utility, temperature=1.0):
    # Calculate the acceptance probability
    if new_utility > old_utility:
        return 1.0
    else:
        #return tf.exp((new_utility - old_utility) / temperature)

def mcmc_optimize(current_matrix, score):
    # Initialize the NxN boolean matrix
    
    current_utility = util(current_matrix)
    for i in range(num_iterations):
        new_matrix = propose_new_matrix(current_matrix)
        new_utility = util(new_matrix)

        # Decide whether to accept the new matrix
        #if tf.random.uniform(()) < acceptance_probability(current_utility, new_utility):
            current_matrix.assign(new_matrix)
            current_utility = new_utility

        # Optionally: Print the current utility or save the matrices for analysis

    return current_matrix

# Example usage
n = 4  # Size of the matrix
optimized_matrix = mcmc_optimization(n)
print("Optimized Matrix:\n", optimized_matrix.numpy())

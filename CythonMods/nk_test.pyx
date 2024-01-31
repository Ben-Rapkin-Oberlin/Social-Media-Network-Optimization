import nkpack as nk
import numpy as np
from scipy.stats import norm
from numpy.typing import NDArray
cimport numpy as cnp
cpdef get_globalmax(imat: NDArray[np.int8], cmat: NDArray[np.float32], n: int):
    """
    Calculate global maximum by calculating performance for every single bit string.
    There is a reason for why it does not save every performance 
    somewhere, so that we can have a giant lookup table and never have to 
    calculate performances ever again, however, the performances are float32 (4 Bytes),
    which means that for 5 agents with 4 tasks each we have (2^20)*4 = 

    Notes:
        Uses Numba's njit compiler.

    Args:
        imat: Interaction matrix
        cmat: Contribution matrix (landscape)
        n: Number of tasks per landscape
        p: Number of landscapes (population size)
    
    Returns:
        The float value with the maximum performance (sum of performance contributions phi[i])

    """

    max_performance = 0.0
    p=1
    for i in range(2 ** (n) ):
        # convert the decimal number i to binary.
        # this long weird function does exactly that very fast 
        # and can be jit-compiled, unlike a more straightforward function.
        # This is equivalent to nk.dec2bin but is inserted here to avoid jit-ing it.
        dec_to_bin = ( (i // 2**np.arange(n)[::-1]) % 2 ).astype(np.int8)

        # calculate performances for p agents
        phis = calculate_performances(dec_to_bin, imat, cmat, n)

        # find global max for aggregate performance
        try:
            if sum(phis) > max_performance:
                max_performance = sum(phis)
        except:
            print('p',phis)
            exit()

    return max_performance




cpdef calculate_performances(bstring: NDArray[np.int8], imat: NDArray[np.int8], cmat: NDArray[np.float32], n: int):
    """
    Computes a performance of a bitstring given contribution matrix (landscape) and interaction matrix

    Notes:
        Uses Numba's njit compiler, so the advanced numpy operations such as np.mean(axis=1)
        are not supported. That is why the code might seem to be dumbed down. But njit
        speeds up any list comprehensions or numpy tricks by 4 times at least
        in this particular case.

    Args:
        x : An input vector
        imat: Interaction matrix
        cmat: Contribution matrix (landscape)
        n: Number of tasks per landscape
        p: Number of landscapes (population size)

    Returns:
        A list of P performances for P agents.

    """
    p=1
    # get performance contributions for every bit:
    phi = np.zeros(n)
    for i in range(n):
        # subset only coupled bits, i.e. where
        # interaction matrix is not zero:
        coupled_bits = bstring[np.where(imat[:,i]>0)]

        # convert coupled_bits to decimal. this long weird function 
        # does exactly that very fast and can be jit-compiled,
        # unlike a more straightforward function. This is equivalent to
        # the function nk.bin2dec but is inserted here to avoid jit-ing it.
        bin_to_dec = sum(coupled_bits * 2**(np.arange(coupled_bits.size)[::-1]))

        # performance contribution of x[i]:
        phi[i] = cmat[bin_to_dec, i] 

    # get agents' performances by averaging their bits'
    # performances, thus getting vector of P mean performances
    Phis = np.zeros(p, dtype=np.float32)
    for i in range(p):
        Phis[i] = phi[n*i : n*(i+1)].mean()
        
    return Phis


cpdef interaction_matrix(N:int, K:int):
    """Creates an interaction matrix for a given K

    Args:
        N: Number of bits
        K: Level of interactions
        shape: Shape of interactions. Takes values 'roll' (default), 'random', 'diag'.

    Returns:
        An NxN numpy array with diagonal values equal to 1 and rowSums=colSums=K+1
    """

    output = None
    if K == 0:
        output = np.eye(N,dtype=int)

    
    tmp = [1]*(K+1) + [0]*(N-K-1)
    tmp = [np.roll(tmp,z) for z in range(N)]
    tmp = np.array(tmp)
    output = tmp.transpose()

    # print(f"Interaction shape '{shape}' selected")
    return output


cpdef generate_landscape(n: int, k: int):
  
    np.random.seed(0)
    base_matrix = np.random.multivariate_normal(mean=[0], cov=[[1]], size=(n*2**(1+k)))
    cdf_matrix = norm.cdf(base_matrix)
    landscape = np.reshape(cdf_matrix.T, (n, (2**(1+k)))).T
    
    return landscape


cpdef all_scores(cnp.ndarray[char, ndim=2] all,cnp.ndarray imat,cnp.ndarray cmat,int n):
    scores=np.zeros(len(all))
    for i in range(len(all)):
        scores[i]=(calculate_performances(all[i],imat,cmat,n)[0])
    return scores



from numpy import zeros, arange, tile, sum
from numpy.random import randint, choice
from collections import Counter

def InterpolStochSMC(generator, discriminator, DoG, config):
    
    """
    
    This function performs interpolation between a pair of codes living in the latent space. The interpolation is stochastic,
    meaning the interpolation is generated by a Gaussian bridge (i.e. a multivariate normal sample conditioned on start and
    end positions). Further, the interpolation utilizes a particle filter that forces the trajectory to move along codes that
    represent realistic samples from the data distribution (in this case images of handwritten digits). The weight function, which
    determines a particle's chance of survival, is given by the discriminator network (a.k.a. 'critic') combined with the
    generator network (a.k.a. 'artist') such that a code may flow through the artist, then the output of which flows through the
    critic and finally becomes a score between 0 and 1. The higher the score, the higher the chance of survival. In this way the
    interpolation is forced to produce a realistic sequence of images in the pixel space.
    
    """
    
    # Input arguments ==============================================================================
    # The 'generator' is a generator network of a GAN (generative adversarial network), here a keras
    # sequential object.
    # The 'discriminator' is a discriminator network of a GAN, also a keras sequential object.
    # The 'DoG' is the combined network of 'generator' and 'discriminator', indeed 'discriminator'
    # composed with 'generator'.
    # The 'config' is the setup of the program set by the user elsewhere.
    
    # Setup ========================================================================================
    # zDim is the dimension of the latent space, data type int64
    # z0 is starting position of interpolation (at time 0), data type numpy.array shape = (zDim, 1)
    # zT is ending position of interpolation (at time T), data type numpy.array shape = (zDim, 1)
    # T is elapsed total time, a tuning paramter essentially, data type int64
    # N is number of visited points of interpolation, including start and end point, data type int64
    # n_parts is size of particle filter, i.e. number of particles, data type int64
    
    zDim = config.z_dim
    z0 = config.z_start
    zT   = confif.z_end
    
    T = config.time
    N = config.steps
    n_parts = config.nmrParts
    
    # Assemble / allocate matrices ====================================================================================
    # weights_all is a matrix where the i:th row, j:th column represents the i:th particle's weight at
    # its j:th step, data type numpy.array shape = (n_parts, N-2)
    # S is a list of indices (from 0 up to (n_parts-1)) for each particle, data type numpy.array shape = (n_parts, )
    # PartsPaths is a 3-D tensor where the i:th sheet represents the i:th particle's entire trajectory from start at z0
    # finish at zT. The j:th row is the j:th coordinate's (there are zDim coordinates) trajectory, and the k:th column
    # is the k:th position of (out of N positions) the particle path, data type numpy.array shape = (n_parts, zDim, N)
    
    weights_all = zeros((n_parts, N-2))
    S = arange(n_parts)
    
    PartsPaths = zeros((1, zDim, N))
    PartsPaths[0,:,0] = z0.reshape(-1)
    PartsPaths[0,:,-1] = zT.reshape(-1)
    PartsPaths = tile(PartsPaths, (n_parts, 1, 1))
    
    # Run program ========================================================================================================
    # dt is the time step used when sampling a Gaussian bridge, data type float64
    # S_re is a list of resampled particle indices, used in the particle filter. It is initialized as a list of zeros,
    # hence pointing at particle with index 0 (note: this does not matter since all particle start at z0), data type
    # numpy.array shape = (n_parts, )
    # weights is a list of weights for each particle's next trajectory step (the i:th element of weights is the associated
    # weight to the particle with index i), data type numpy.array shape = (n_parts, )
    
    dt = T / N
    S_re = randint(0, 1, n_parts)
    weights = zeros(n_parts)
    
    for step in range(N-2):
        
        print('>' * 4 + 'Remaining steps: ', (N-2)-step, '<' * 4)
        
        # Outer for-loop ==================================================================================================
        # The idea is to generate n_parts Gaussian bridges between z0 and zT where z0 (the starting position) is updated
        # at the end of every loop. Indeed, the Gaussian bridges' duration / time length and number of time steps decreases
        # linearly for every loop such that the time step is held constant.
        
        # Tcur is the remaining time, data type float64
        # Ncur is the remaining number of time steps that should be taken, data type int64
        # parts is a 3-D tensor that stores the generated Gaussian bridges for each of the n_parts particles. Note: number of
        # columns (= Ncur) decreases with each loop, data type numpy.array shape = (n_parts, zDim, Ncur)
        
        Tcur = T - dt * step
        Ncur = N - step
        parts = zeros((n_parts, zDim, Ncur))
        
        # surv_freq, short for 'survivor-frequency', is a dictionary constructed from S_re where each key is an index (survivor
        # of the resampling step) and the corresponding value is the frequency that index appeared in S_re, data type python
        # dictionary object
        # resampled is a list of 'survivor' indices, data type numpy.array shape = (_, )
        # freq is a list of frequencies for resampled, data type numpy.array shape = (_, )
        
        surv_freq = Counter(S_re)
        resampled = list(surv_freq.keys())
        freq = list(surv_freq.values())
        
        # First inner for-loop ================================================================================================
        # Here we evolve the particles according to their new positions (after the resampling step below). The new positions
        # are the updated z0 as explained above. Note that the particles may have different z0 which is why iteration over all
        # different such z0 is necessary. Further, observe that several particles may have the same z0, which is also taken into
        # consideration.
        # The BPvec function (short for Bridge-Process-vectorized) outputs f Gaussian bridges with start and end points z0 and
        # zT, respectively. The time constraint is set by Tcur and number of steps by Ncur.
        
        # pointer is just keeping track of how many particles we have evolved so far, data type int64
        # nmr_res is the number of unique resampled particle indices, data type int64
        
        pointer = 0
        nmr_res = len(resampled)
        
        for i in range(nmr_res):
            
            res_idx = resampled[i]
            f = freq[i]
            
            z0 = PartsPaths[res_idx,:,step].reshape(zDim,-1)
            parts[pointer:pointer+f,:,:] = BPvec(z0, zT, Tcur, Ncur, f)
            
            pointer += f
        
        # Second inner for-loop ================================================================================================
        # Here we perform the resampling step of the particle filter. Note that the 3-D tensor parts contains proposed
        # particle paths, however only the second position of each particle path, i.e. the second column of each sheet in parts,
        # is of interest (since we evolve the particle filter stepwise). Each such proposed next-step is given a weight assigned
        # by the function weight_func (which in essence is the combined DoG network).
        
        for idx in range(n_parts):
            
            z = parts[idx,:,1] 
            weights[idx] = weight_func(z, DoG)
        
        # Normalize weights
        weights = weights / sum(weights)
        
        # Resampling
        S_re = choice(S, n_parts, replace=True, p=weights)
        parts[:,:,1:] = parts[S_re,:,1:]
        
        # Storing weights for each new position of every particle path
        weights_all[:,step] = weights[S_re]

        # Update particle paths
        PartsPaths[:,:,step+1] = parts[:,:,1]
    
    print('>' * 4 + 'Finished' + '<' * 4)
    
    # Return the interpolation ============================================================================================
    # The interpolation that will be returned is a sequence of the positions with highest weigthts for every step over all
    # the particles. In other words, for the j:th column in weights_all we select the row index i with largest compoent / 
    # weight, implying that the j:th column of the i:th sheet of PartsPaths will be selected as the j:th position in the
    # interpolation.
    # The explicit function simply returns indices corresponding to the maximum value of a list.
    
    # interpol is initialized as 0:th sheet of PartsPaths (this way the interpolation contains the correct start and end points),
    # data type numpy.array shape = (zDim, N)
    interpol = PartsPaths[0,:,:]
    
    for step in range(N-2):
        
        idxs, _ = explicit(weights_all[:,step])
        idx = idxs[0][0]
        
        interpol[:,step+1] = PartsPaths[idx,:,step+1]
        
    return interpol


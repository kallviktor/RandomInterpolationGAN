#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy import exp, abs, linspace, zeros, array, ones, kron, where, eye
from numpy.random import multivariate_normal

def ker(h, a=1, b=1):
    
    """ 
    Eq. (9) in Hult et al.
    This is the kernel of the Gaussian process.
    """
    
    return exp(-b * abs(h) ** a)


def muhat_mean(t, z0, zT, T):
    
    """
    Eq. (12) in Hult et al.
    Mean function for Gaussian bridge with kernel function 'ker'.
    """
    
    # Input arguments ================================================================================================
    # t is a discretized, equidistant time interval, data type numpy.array shape = (1, _)
    # z0 is the start position, data type numpy.array shape = (zDim, 1)
    # zT is the end position, data type numpy.array shape = (zDim, 1)
    # T is the total time, data type int64
    
    # Idea ===========================================================================================================
    # This function outputs a matrix where the i:th column corresponds to the mean of the i:th coordinate's Gaussian 
    # bridge.
    
    
    num   = z0 * (ker(t) - ker(T-t) * ker(T)) + zT * (ker(T-t) - ker(t) * ker(T))
    denom = 1 - ker(T) ** 2
    
    return num / denom


def khat_cov(tmat, smat, T):
    
    """
    Eq. (12) in Hult et al.
    Covariance function for Gaussian bridge with kernel function 'ker'.
    """
    
    # Input arguments ================================================================================================
    # tmat is a matrix where the rows are constant and where each column represents equidistant time points, data type
    # numpy.array shape = (_, _)
    # smat is a matrix where the columns are constant and where each row represents equidistant time points. smat is the
    # transpose of tmat, data type numpy.array shape = (_, _)
    # T is the total time, data type int64
    
    # Idea ===========================================================================================================
    # This function outputs a covariance matrix with kernel function given in eq. (12) in Hult et al. This covariance
    # matrix is the same for all zDim coordinate processes.
    
    T1 = ker(tmat-smat)
    
    num = ker(T) * (ker(T-smat) * ker(tmat) + ker(smat) * ker(T-tmat)) - ker(T-smat) * ker(T-tmat) - ker(smat) * ker(tmat)
    denom = 1 - ker(T) ** 2
    T2 = num / denom
    
    return T1 + T2


def BPvec(z0, zT, T, N, nParticles):
    
    """
    This function generates several Gaussian bridge processes. 
    """
    
    # Input arguments ================================================================================================
    # z0 is the start position, data type numpy.array shape = (zDim, 1)
    # zT is the end position, data type numpy.array shape = (zDim, 1)
    # T is the total time, data type int64
    # N is the number time steps, data type int 64
    # nParticles is essentially the batch size, data type int 64
    
    # Assemble matrices of time differences ==========================================================================
    # t is a uniform discretization of the time interval, data type numpy.array shape = (1, N)
    # smat is a matrix where each column has constant entries, data type numpy.array shape = (N, N)
    # tmat is the transpose of smat, data type numpy.array shape = (N, N)

    t = array([linspace(0, T, N)])
    smat = tile(t, (3, 1))
    tmat = smat.transpose(1,0)

    # Generate multi-dimensional random walk / Gaussian bridge ========================================================
    
    # cov is a covariance matrix (identical for all coordinate processes), data type numpy.array shape = (N, N)
    # block_cov is 100N-by-100N diagonal block matrix with cov as its diagonal entries, data type numpy.array
    # shape = (100N, 100N)
    
    cov = khat_cov(tmat, smat, T)
    block_cov = kron(eye(dim), cov)

    # means is a collection of mean vectors (one mean vector / column for each coordinate process), data type numpy.array
    # shape = (zDim, N)
    # flat_means transforms means to a column vector (each column in means staked on each other), data type numpy.array
    # shape = (zDim * N, 1)
    
    means = muhat_mean(t, z0, zT, T)
    flat_means = means.ravel()
    
    # Sampling nParticles Gaussian bridges at once
    
    BatchGB = multivariate_normal(flat_means, cov=block_cov, size=nParticles)
    BatchGB = out.reshape((-1, ) + means.shape)
    
    # Output BatchGB is a 3-D tensor where each sheet is a Gaussian bridge corresponding to a particle, data type numpy.array
    # shape = (nParticles, zDim, N)
    
    return BatchGB

def weight_func(z, DoG, config):
    
    """
    The weight function for the particle filter. This is the critic / discriminator network's guess at how realistic the
    generated image (with code z) is.
    """
    
    z = z.reshape(-1,100)

    D_x = DoG.predict(z)[0,0]

    weight = D_x/(1 - D_x)
    return weight

def explicit(l):
    max_val = max(l)
    max_idx = np.where(l == max_val)

    return max_idx, max_val


# In[5]:


from numpy import array, ones, linspace, tile

T = 1
N = 3

t = array([linspace(0, T, N)])
I = ones((N, N))
tmat = t*I
tmat = tmat.T


# In[6]:


print(tmat)


# In[ ]:





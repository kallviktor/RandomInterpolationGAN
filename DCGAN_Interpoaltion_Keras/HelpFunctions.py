from numpy import exp, abs,

def ker(h, a=1, b=1):
    
    """ 
    Eq. (9) in Hult et al.
    This is the kernel
    """
    
    return exp(-b * abs(h) ** a)


def muhat(t, z0, zT, T):
    
    num   = z0 * (ker(t) - ker(T-t) * ker(T)) + zT * (ker(T-t) - ker(t) * ker(T))
    denom = 1 - ker(T) ** 2
    
    return num / denom


def khat(t, s, T):
    
    T1    = ker(t-s)
    
    num   = ker(T) * (ker(T-s) * ker(t) + ker(s) * ker(T-t)) - ker(T-s) * ker(T-t) - ker(s) * ker(t)
    denom = 1 - ker(T) ** 2
    T2    = num / denom
    
    return T1 + T2


def khatvec(tmat, smat, T):
    
    T1 = ker(tmat-smat)
    
    num = ker(T) * (ker(T-smat) * ker(tmat) + ker(smat) * ker(T-tmat)) - ker(T-smat) * ker(T-tmat) - ker(smat) * ker(tmat)
    denom = 1 - ker(T) ** 2
    T2 = num / denom
    
    return T1 + T2


def BPvec(z0, zT, T, N, nParticles):
    # Bridge process from z0 to zT in N steps (over time-dimension) and in time T

    t_grid = np.linspace(0, T, N)    # Uniform grid on t-axis

    dim = len(z0)
    Z = np.zeros((dim,N))

    # Generate covariance matrix (identical for all coordinate processes!)
    # These steps here work in practice, but need furnishing. It is not obvious what happens here really!

    t_grid = np.array([np.linspace(0, T, N)])
    I = np.ones((N,N))
    t = t_grid
    tmat = t*I
    tmat = tmat.T
    s = t
    smat = s*I

    # Generate covariance matrix
    cov = khatvec(tmat, smat, T)

    means = np.zeros((dim,N))
    means = muhat(t, z0, zT, T)   # Dimension dim-by-N

    # Generate collection of mean vectors (for each coordinate process)
    means = np.zeros((dim,N))
    means = muhat(t, z0, zT, T)   # Dimension dim-by-N

    # Generate multi-dimensional random walk (one RW for each coordinate according to its mean and the covariance matrix)
    num_samples = nParticles
    flat_means = means.ravel()

    # Vectorized sampling from several multivariate normal distributions at once
    # Build block covariance matrix
    block_cov = np.kron(np.eye(dim), cov)
    out = np.random.multivariate_normal(flat_means, cov=block_cov, size=num_samples)
    out = out.reshape((-1,) + means.shape)
    # Batch of random bridge walks (this is a collection of matrices where each matrix is a multidimensional random walk)
    # The size of the batch is given by the input argument nParticles
    batchRWs = out

    return batchRWs

def weight_func(z, dcgan, config):
    # print('Shape z: ', z.shape)
    z = z.reshape(-1,100)

    D_x = dcgan.combined.predict(z)[0,0]

    weight = D_x/(1 - D_x)
    return weight

def explicit(l):
    max_val = max(l)
    max_idx = np.where(l == max_val)
    # max_idx = l.index(max_val)
    return max_idx, max_val


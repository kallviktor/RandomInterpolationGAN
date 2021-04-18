from numpy import exp, abs, linspace, zeros, array, ones, matmul, kron, where, eye, tile, meshgrid, concatenate, arange, round, block
from numpy.random import multivariate_normal, normal
from numpy.linalg import inv
import numpy as np
import os
import time
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def ker(h, a=2, b=5):
    #2,5
    
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
"""
from interpolations_help_fcns import ker, khat_cov, muhat_mean
from numpy import linspace, array, zeros, tile, matmul, kron, eye, ravel, ones
from numpy.linalg import inv

T = 1
N = 5

z0 = array([[0],[0]])
zT = array([[1],[1]])

zdim = z0.shape[0]

def linear_interpol(start, end, N):
    # Linear interpolation
    t     = array(linspace(0, 1, N))
    z_seq = end * t + (1 - t) * start
    return z_seq

path = linear_interpol(z0, zT, N).T

Delta = T / (N-1)
times = linspace(0,T,N)

times = array([times])

smat = tile(times, (N, 1))
tmat = smat.transpose(1,0)

Sigma = khat_cov(smat, tmat, T)
Mu = muhat_mean(times, z0, zT, T)

# Goal: for a process (z{1} , z{2} , ... , z{dim}) running over N steps, starting in z0 and ending in zT, construct the mean for every coordinate process z{i}
# and with a time horizon T

MeanMatrix = muhat_mean(times, z0, zT, T).T

# Column i in MeanMatrix is the mean of coordinate process z{i}, where z{i} is a bridge between z0{i} to zT{i}

Sigma = khat_cov(smat, tmat, T)
SigmaBlock = kron(eye(zdim), Sigma)

# SigmaBlock is blockdiagonal matrix with Sigma, which is the covariance matrix of z{i}, on the diagonal. Note that Sigma is the same for each z{i}, so it
# does in fact not depend on z0{i} and zT{i} (which is the difference between coordinate processes)

# New goal: find expression of joint distribution for (z0 , z1 , z2 , ... , zi , zT), where zj = (zj{1} , zj{2} , ... , zj{dim}) with j = step and
# 1 <= i <= N-1. Here we use notation zT = zN.
"""
def NewjointCov(Sigma, step):

    SigmaCur = zeros((step,step))

    SigmaCur[0:step,0:step] = Sigma[1:step+1,1:step+1]
    """
    SigmaCur[-1,0:step+1] = Sigma[-1,0:step+1]
    SigmaCur[0:step+1,-1] = Sigma[0:step+1,-1]
    SigmaCur[-1,-1] = Sigma[-1,-1]

    SigmaCur[[0,step],:] = SigmaCur[[step,0],:]
    SigmaCur[:,[0,step]] = SigmaCur[:,[step,0]]
    """

    SigmaCur = np.flip(SigmaCur, 0)
    SigmaCur = np.flip(SigmaCur, 1)

    return SigmaCur
"""
i = 2

SigmaNew = NewjointCov(Sigma, i)
SigmaNewBlock = kron(eye(zdim), SigmaNew)
"""
# Here we have also rearanged the vector (z0 , z1 , z2 , ... , zi , zT) -> (zi , z1 , z2 , ... , z0 , zT). 

# Newer goal: do the same for the mean. That is, what is the mean for (zi , z1 , z2 , ... , z0 , zT)?

def NewjointMean(Mean, step, zdim):

    MuCur = zeros((step,zdim))
    
    MuCur[0:step,:] = Mean[1:step+1,:]

    """
    MuCur[step+1,:] = Mean[-1,:]
    MuCur[[0,step],:] = MuCur[[step,0],:]

    """

    MuCur = np.flip(MuCur,0)

    return MuCur
"""
MeanMatrixNew = NewjointMean(MeanMatrix, i, zdim)
"""
# Ultimate goal: find the condtional distribution of Zi | Z1 , Z2 , ... , Z0 , ZT. The standard formulas give us the answer, however it is a little tricky
# to make it all happen at once (without for-loops)

# First, what is the conditonal mean? Let's go:

def MeanConditional(means, sigma, path, step, zdim):

    #path[[0,step],:] = path[[step,0],:]
    #print('path: ',path)

    path = path[1:step,:]

    #print(path)

    path = np.flip(path,0)

    #print(path)

    Zhist = zeros((step-1,zdim))

    Zhist = path.T
    #Zhist[step,:] = path[-1,:]
    Zhist = Zhist.ravel().T

    #print(sigma[1:,1:])

    k22Inv = inv(sigma[1:,1:])

    #print(k22Inv)

    k22InvBlock = kron(eye(zdim),k22Inv)

    k12 = sigma[0,1:]

    k12Block = kron(eye(zdim),k12)

    m1 = means[0,:].T

    #print("m1 shape= " , m1.shape)


    m2 = means[1:,:].T
    m2 = m2.ravel().T


    #print(m2)
    #print('hist: ',Zhist)
    #print('')
    #print(matmul(k12Block,k22InvBlock))


    return m1 + matmul(matmul(k12Block,k22InvBlock),Zhist-m2)

"""
MeanCond = MeanConditional(MeanMatrixNew, SigmaNew, path, i, zdim)
"""
# Also, find the conditional covariance:



def CovConditional(sigma, zdim):

    k11 = sigma[0,0]
    k12 = sigma[0,1:]
    k22Inv = inv(sigma[1:,1:])
    k21 = sigma[1:,0]

    std = k11 - matmul(matmul(k12,k22Inv),k21)

    return kron(eye(zdim),std)
"""
CovCond = CovConditional(SigmaNew, zdim)

print(MeanCond)

print(CovCond)
"""

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

    z_dim = len(z0)
    t = array([linspace(0, T, N)])
    smat = tile(t, (N, 1))
    tmat = smat.transpose(1,0)

    # Generate multi-dimensional random walk / Gaussian bridge ========================================================
    
    # cov is a covariance matrix (identical for all coordinate processes), data type numpy.array shape = (N, N)
    # block_cov is 100N-by-100N diagonal block matrix with cov as its diagonal entries, data type numpy.array
    # shape = (100N, 100N)
    
    cov = khat_cov(tmat, smat, T)
    block_cov = kron(eye(z_dim), cov)

    # means is a collection of mean vectors (one mean vector / row for each coordinate process), data type numpy.array
    # shape = (zDim, N)
    # flat_means transforms means to a column vector (each column in means staked on each other), data type numpy.array
    # shape = (zDim * N, 1)
    
    means = muhat_mean(t, z0, zT, T)
    flat_means = means.ravel()
    
    # Sampling nParticles Gaussian bridges at once
    
    BatchGB = multivariate_normal(flat_means, cov=block_cov, size=nParticles)
    BatchGB = BatchGB.reshape((-1, ) + means.shape)
    
    # Output BatchGB is a 3-D tensor where each sheet is a Gaussian bridge corresponding to a particle, data type numpy.array
    # shape = (nParticles, zDim, N)
    
    return BatchGB

def track(zvec):

    weights = zeros(zvec.shape[0])
    i = 0
    for z in zvec:
        if (z[0] > 0.45 and z[0] < 0.55 and z[1] > -1.05 and z[1] < 0.55) or (z[0] > -0.55 and z[0] < 0.45 and z[1] < -0.95 and z[1] > -1.05):
            weights[i] = 0.95
        else:
            weights[i] = 0
        i += 1
    #print(weights)
    return weights

def weight_func(z, z_dim, DoG):
    
    """
    The weight function for the particle filter. This is the critic / discriminator network's guess at how realistic the
    generated image (with code z) is.
    """

    z = z.reshape(-1,z_dim)

    Dz = DoG.predict(z).reshape(-1)

    #Dz = np.zeros(len(z))*0+0.5
    #Dz = Dz.T

    #Dz = track(z)
    #Dz[Dz<0.3]=0



    weights = (Dz / (1 - Dz))**1


    
    return weights

def explicit(l):
    max_val = max(l)
    max_idx = where(l == max_val)

    return max_idx, max_val

def vis_interpolation(config,G,paths):
    
    if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    if not os.path.exists(config.inter_dir):
        os.makedirs(config.inter_dir)

    i = 0
    for key in config.interpol_types.keys():
        for j in range(config.interpol_types[key]):

            path = paths[i,:,:].T
            fake_imgs = G.predict(path)
            #fake_imgs = 0.5 * fake_imgs + 0.5

            fig,axs = plt.subplots(1,config.int_steps)
            count = 0

            for k in range(config.int_steps):

                axs[k].imshow(fake_imgs[count,:,:,0], cmap='gray')
                axs[k].axis('off')
                count += 1

            #Check if filename exists and if so, save with same but new ending.
            filename = '/inter_{}_{}steps_{}t'.format(key,config.int_steps,str(config.int_time).replace('.',''))

            filepath = dynamic_filepath(config.inter_dir,filename)

            plt.savefig(filepath)  
            plt.close()
            i += 1

def dynamic_filepath(save_dir,filename):
    #Check if filename exists and if so, save with same but new ending.
    #filename = '/inter_{}steps_{}t'.format(config.int_steps,str(config.int_time).replace('.',''))
    filepath = save_dir+filename
    existing_filepaths = glob.glob(filepath+'*')

    if existing_filepaths:
        #sort and find last file among existing files with same filename. [-5] due to .png ending.
        existing_filepaths = sorted(existing_filepaths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        last_filepath = existing_filepaths[-1]
        #find ending of last file. [-5] due to .png ending.
        last_ending = last_filepath.split('_')[-1].split('.')[0]
        new_filepath = filepath+'_{}'.format(int(last_ending)+1)
    else:
        new_filepath = filepath+'_1'

    return new_filepath

def print_interpolation_initialized():
    print('\n' * 1)
    print('='*65)
    print('-'*19,'Interpolation initialized','-'*19)
    print('='*65)
    print('\n' * 1)

def print_interpolation_complete():
    print('\n' * 1)
    print('='*65)
    print('-'*20,'Interpolation complete','-'*21)
    print('='*65)
    print('\n' * 1)

def print_interpolation_progress(N,step):
    print('Step {}/{}'.format(step+1,N-2))

def get_valid_code(DoG, config):

    threshold = config.thresh
    while True:

        z   = normal(0, 1, size=(1,config.z_dim)) 
        score = DoG.predict(z)

        if score > threshold:
            return z.T

def heat_map(DoG, config):

    if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    if not os.path.exists(config.hm_dir):
        os.makedirs(config.hm_dir)

    xmin = config.hm_xmin
    xmax = config.hm_xmax
    ymin = config.hm_ymin
    ymax = config.hm_ymax
    steps = config.hm_steps

    if steps % 2 == 0:
        steps += 1

    skip = int((steps-1)/10)


    z1 = linspace(xmin, xmax, steps)
    z2 = linspace(ymin, ymax, steps)

    zx, zy = meshgrid(z1, z2)

    zx = zx.reshape(1,-1)
    zy = zy.reshape(1,-1)

    zbatch = concatenate((zx, zy)).T

    scores = DoG.predict(zbatch).reshape(steps,steps)

    plt.figure(figsize=(10, 10))

    im = plt.imshow(scores, cmap='hot',extent=[0,steps,0,steps])
    

    plt.clim(0, 1)
    plt.colorbar(im,fraction=0.046, pad=0.04,orientation='horizontal')
    plt.xticks(arange(0, steps, skip),round(z1[0::skip],1),rotation='horizontal')
    plt.yticks(arange(0, steps, skip),round(z2[0::skip],1))

    locs, labels = plt.yticks()

    filename = '/heatmap_{}'.format(config.hm_steps)
    filepath = dynamic_filepath(config.hm_dir,filename)
    plt.savefig(filepath)
    plt.close()

def latent_visualization(config,G):

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    if not os.path.exists(config.latvis_dir):
        os.makedirs(config.latvis_dir)

    xmin = config.hm_xmin
    xmax = config.hm_xmax
    ymin = config.hm_ymin
    ymax = config.hm_ymax
    steps = config.hm_steps

    skip = int((steps-1)/10)


    z1 = linspace(xmin, xmax, steps)
    z2 = linspace(ymax, ymin, steps)

    zx, zy = meshgrid(z1, z2)

    zx = zx.reshape(1,-1)
    zy = zy.reshape(1,-1)

    zbatch = concatenate((zx, zy)).T

    imgs = G.predict(zbatch).reshape(steps, steps, 32, 32)


    plt.figure(figsize=(10, 10))
    plt.imshow(block(list(map(list, imgs))),cmap='gray',extent=[xmin,xmax,ymin,ymax])

    filename = '/latent_vis_{}'.format(config.hm_steps)
    filepath = dynamic_filepath(config.latvis_dir,filename)
    plt.savefig(filepath)
    plt.close()

def latent_inter(config, paths, G):

    if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    if not os.path.exists(config.inter_dir):
        os.makedirs(config.inter_dir)

    xmin = config.hm_xmin
    xmax = config.hm_xmax
    ymin = config.hm_ymin
    ymax = config.hm_ymax
    steps = config.hm_steps

    if steps % 2 == 0:
        steps += 1

    skip = int((steps-1)/10)


    z1 = linspace(xmin, xmax, steps)
    z2 = linspace(ymax, ymin, steps)

    zx, zy = meshgrid(z1, z2)

    zx = zx.reshape(1,-1)
    zy = zy.reshape(1,-1)

    zbatch = concatenate((zx, zy)).T


    imgs = G.predict(zbatch).reshape(steps, steps, 32, 32)

    plt.figure(figsize=(10, 10))
    plt.imshow(block(list(map(list, imgs))),cmap='gray',extent=[xmin,xmax,ymin,ymax])

    
    i = 0
    for key in config.interpol_types.keys():
        for j in range(config.interpol_types[key]):

            path = paths[i,:,:].T

            x = path[:,0]
            y = path[:,1]
            
            if key == 'linear':
                color = 'b'
            elif key == 'stoch':
                color = 'r'
            elif key == 'stochSMC':
                color = 'y'
            
            """
            if j == 0:
                color = 'b'
            elif j == 1:
                color = 'r'
            elif j == 2:
                color = 'y'
            """
            plt.plot(x, y, linestyle='--', marker='o', color=color)

            i += 1
    filename = '/inter_latent_vis'
    filepath = dynamic_filepath(config.inter_dir,filename)
    plt.savefig(filepath)
    plt.close()





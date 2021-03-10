from numpy import exp, abs, linspace, zeros, array, ones, kron, where, eye, tile, meshgrid, concatenate, arange, round, block
from numpy.random import multivariate_normal, normal
import numpy as np
import os
import time
import glob
import matplotlib.pyplot as plt


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

def weight_func(z, z_dim, DoG):
    
    """
    The weight function for the particle filter. This is the critic / discriminator network's guess at how realistic the
    generated image (with code z) is.
    """

    z = z.reshape(-1,z_dim)

    Dz = DoG.predict(z).reshape(-1)

    weights = (Dz / (1 - Dz))**1
    
    return weights

def explicit(l):
    max_val = max(l)
    max_idx = where(l == max_val)

    return max_idx, max_val

def vis_interpolation(config,G,path):
    
    if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    if not os.path.exists(config.inter_dir):
        os.makedirs(config.inter_dir)

    #path = path.reshape(-1,config.z_dim)
    path = path.T
    fake_imgs = G.predict(path)
    #fake_imgs = 0.5 * fake_imgs + 0.5

    fig,axs = plt.subplots(1,config.int_steps)
    count = 0
    for i in range(config.int_steps):

        axs[i].imshow(fake_imgs[count,:,:,0], cmap='gray')
        axs[i].axis('off')
        count += 1

    #Check if filename exists and if so, save with same but new ending.
    filename = '/inter_{}steps_{}t'.format(config.int_steps,str(config.int_time).replace('.',''))
    filepath = config.inter_dir+filename
    existing_filepaths = glob.glob(filepath+'*')

    if existing_filepaths:
        #sort and find last file among existing files with same filename. [-5] due to .png ending.
        existing_filepaths = sorted(existing_filepaths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        last_filepath = existing_filepaths[-1]
        #find ending of last file. [-5] due to .png ending.
        last_ending = last_filepath.split('_')[-1].split('.')[0]
        plt.savefig(filepath+'_{}'.format(int(last_ending)+1))
    else:
        plt.savefig(filepath+'_1')
        
    plt.close()
def dynamic_filename(filename):
    pass
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
    im = plt.imshow(scores, cmap='hot')

    plt.colorbar(im)
    plt.xticks(arange(0, steps, skip),round(z1[0::skip],1),rotation='vertical')
    plt.yticks(arange(0, steps, skip),round(z2[0::skip],1))

    plt.savefig(config.hm_dir+'/heatmap_2')
    plt.close()

def latent_visualization(config,G):

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
    #print(zbatch.shape)
    #z_grid = np.dstack(np.meshgrid(z1, z2))

    #x_pred_grid = vae.generate(z_grid.reshape(m*m, 2),False).numpy().reshape(m, m, 32, 32)

    imgs = G.predict(zbatch).reshape(steps, steps, 32, 32)
    #print(imgs.shape)
    #print(block(list(map(list, imgs))).shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(block(list(map(list, imgs))),origin='lover',cmap='gray',extent=[xmin,xmax,ymin,ymax])

    plt.savefig(config.hm_dir+'/latent_vis_2')
    plt.close()

def latent_inter(config, path, G):

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
    #print(zbatch.shape)
    #z_grid = np.dstack(np.meshgrid(z1, z2))

    #x_pred_grid = vae.generate(z_grid.reshape(m*m, 2),False).numpy().reshape(m, m, 32, 32)

    imgs = G.predict(zbatch).reshape(steps, steps, 32, 32)
    #print(imgs.shape)
    #print(block(list(map(list, imgs))).shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(block(list(map(list, imgs))),origin='lover',cmap='gray',extent=[xmin,xmax,ymin,ymax])

    
    x = path[0,:]
    y = path[1,:]
    plt.plot(x, y, linestyle='--', marker='o', color='b')
    plt.savefig(config.inter_dir+'/inter_latent_vis_2')
    plt.close()





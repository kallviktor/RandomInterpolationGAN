import numpy as np
from utils import save_images
import math
import os

def ker(h, a=1, b=1):
  """ 
  Eq. (9) in Hult et al
  """
  return np.exp(-b*np.abs(h)**a)


def muhat(t, z0, zT, T):
  num = z0*(ker(t) - ker(T-t)*ker(T)) + zT*(ker(T-t) - ker(t)*ker(T))
  denom = 1 - ker(T)**2
  return num/denom


def khat(t, s, T):
  T1 = ker(t-s)
  num = ker(T)*(ker(T-s)*ker(t) + ker(s)*ker(T-t)) - ker(T-s)*ker(T-t) - ker(s)*ker(t)
  denom = 1 - ker(T)**2
  T2 = num/denom
  return T1 + T2

def khatvec(tmat, smat, T):
  T1 = ker(tmat-smat)
  num = ker(T)*(ker(T-smat)*ker(tmat) + ker(smat)*ker(T-tmat)) - ker(T-smat)*ker(T-tmat) - ker(smat)*ker(tmat)
  denom = 1 - ker(T)**2
  T2 = num/denom
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

def BP(z0, zT, T, N):
  # Bridge process from z0 to zT in N steps (over time-dimension) and in time T
  
  t_grid = np.linspace(0, T, N)    # Uniform grid on t-axis

  # For each coordinate of z0 and zT we simulate a bridge-process

  dim = len(z0)
  Z = np.zeros((dim,N))

  for (i, z0_i, zT_i) in zip(range(dim), z0, zT):
    mean_i = np.zeros(N)    # mean.shape --> (N,) i.e. a column vector of length N
    for k in range(N):
      t = t_grid[k]
      mean_i[k] = muhat(t, z0_i, zT_i, T)
    
    cov_i = np.zeros((N,N))
    for m in range(N):
      t = t_grid[m]
      for n in range(N):
        s = t_grid[n]
        cov_i[m][n] = khat(t, s, T)
    
    Z_i = np.random.multivariate_normal(mean_i, cov_i)
    Z[i][:] = Z_i
  Z.shape
  return Z

def gen_particles(z0, zT, T, N, batch_size):
  samples = np.zeros(shape=(dim,N,batch_size))
  pr = 0
  for i in range(batch_size):
    z0_i = z0[:,i]
    zT_i = zT[:,i]
    samples[:,:,i] = BP(z0_i, zT_i, T, N)
    if (i/batch_size) >= pr:
      percent = round(pr*100)
      pr += 0.1
      print('Percentage finished: {}%'.format(percent))
  print('Percentage finished: 100%')
  print('Done!')

  return samples[:,1,:]

def weight_func(z, sess, dcgan, FLAGS):
  z = z[np.newaxis]
  #x = G(z)

  y = np.random.choice(10, FLAGS.batch_size)
  y_one_hot = np.zeros((FLAGS.batch_size, 10))
  y_one_hot[np.arange(FLAGS.batch_size), y] = 1

  x = sess.run(dcgan.sampler, feed_dict={dcgan.z: z, dcgan.y: y_one_hot})
  D_x = sess.run(dcgan.D, feed_dict={dcgan.inputs: x, dcgan.y: y_one_hot})

  weight = D_x/(1 - D_x)
  return weight

def interpolation_linear():
  pass

def visualizePath(Z, dcgan, sess, N, FLAGS):
  image_frame_dim = int(math.ceil(FLAGS.batch_size**.5))
  y = np.random.choice(10, N)
  y_one_hot = np.zeros((N, 10))
  y_one_hot[np.arange(N), y] = 1
  print('y_hot:  ', y_one_hot.shape)
  print('')
  print('Z_shape:  ',Z.shape)
  for i in range(N):
    z = Z[:,i][np.newaxis]
    yy = y_one_hot[i][np.newaxis]
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z})#, dcgan.y: yy})   
    save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(FLAGS.sample_dir, 'test_arange_%s.png' % str(i+2)))


def interpolation_stochastic(sess, dcgan, FLAGS):

  zDim = FLAGS.z_dim
  #zDim = 2

  z0 = np.zeros((zDim,1))    # z0.shape --> (100,)
  zT = np.ones((zDim,1))   # zT.shape --> (100,)

  T = 1
  N = 10    # Number of steps of random walk / sample of process in t-dimension (time-dimension)

  n_parts = 50
  parts = np.zeros((n_parts,zDim,N))

  # En idé för att få algoritmen att gå snabbare
  mean_path = np.zeros((zDim,N))
  mean_path[:,0] = z0.T
  mean_path[:,-1] = zT.T

  S = np.arange(n_parts)    # Set of indices, just in resampling step below

  dt = T/N    # Length of timestep
  for step in range(N-2):
    Tcur = T - dt*step
    Ncur = N - step
    parts = np.zeros((n_parts,zDim,Ncur))
    weights = np.zeros(n_parts)
    parts = BPvec(z0, zT, Tcur, Ncur, n_parts)
    for k in range(n_parts):
      z = parts[k,:,1]    # The first step / forward step is only of importance
      w = weight_func(z, sess, dcgan, FLAGS)
      weights[k] = w
    weights = weights/np.sum(weights)   # Normalize weights
    # Resampling
    S_re = np.random.choice(S, n_parts, replace=True, p=weights)   # Resampling indices in S rather than data points z directly
    parts = parts[S_re,:,:]

    # Save steps taken
    mean_path[:,step+1] = np.mean(parts[:,:,1],0)[np.newaxis]
  visualizePath(mean_path, dcgan, sess, N, FLAGS)
  print('Success')

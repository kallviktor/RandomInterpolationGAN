"""This is the main file used to run the program. 
Change values in the model_config input in order to change setup."""

import model
from interpolation import stochasticSMC_interpol, linear_interpol, stochastic_interpol
from interpolations_help_fcns import vis_interpolation, heat_map, latent_visualization, latent_visualization_64, latent_inter, NewjointCov, get_valid_code
from setup import model_config
from model_help_fcns import *
from google_help_fcns import *
from model_help_fcns import save_model, load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf


#prevent INFO and WARNING messages from printing
#	0 = all messages are logged (default behavior)
#	1 = INFO messages are not printed
#	2 = INFO and WARNING messages are not printed
#	3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#create configuration object
config = model_config(dataset='lines',
					  loadmodel=True,
					  interpolation=True,
					  metrics=False,
					  heat_map=False,
					  latent_viz=False,
					  metrics_k=50,
					  metrics_type='stoch',
					  interpol_types={'stochSMC':1,'stoch':1,'linear':1},
					  thresh=0.1,
					  epochs=20,
					  batch_size=64,
					  lines_batches=100,
					  z_dim=64,
					  z_start=0,
					  z_end=1,
					  int_time=1,
					  int_steps=16,
					  nmrParts=500,
					  gf_dim=8,
					  gfc_dim=128*2,
					  dfc_dim=64,
					  c_dim=1,
					  optimizer='Adam',
					  loss_f='Goodfellow',
					  learning_rate= 0.0002,
					  beta_1=0.5,
					  init='RandomNormal',
					  init_stddev=0.02,
					  clip=0.01,
					  n_critic=5,
					  progress_freq=10,
					  vis_freq=10,
					  plottrain_freq=10,
					  random_sample=False,
					  concatenate=True,
					  hm_xmin=-8,
					  hm_xmax=8,
					  hm_ymin=-8,
					  hm_ymax=8,
					  hm_steps=31,
					  out_dir='/out',
					  load_dir=r'/Users/erikpiscator/Repositories/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210419-1910_lines/models/models_8ep')


if config.loadmodel:

	print_loading_initialized()
	D = load_model(config,'D_')
	G = load_model(config,'G_')
	GAN = load_model(config,'GAN_')
	print_loading_complete()
	
	"""
	print('G')
	print(G.summary())
	print('D')
	print(D.summary())
	"""

else:

	print_training_setup(config)
	dcgan = model.dcgan(config)
	#print(dcgan.G.summary())
	#print(dcgan.D.summary())
	dcgan.train(config)

if config.interpolation:

	nmr_interpols = sum(config.interpol_types.values())
	paths = np.zeros((nmr_interpols,config.z_dim,config.int_steps))

	#z0 = np.array([[],[]])
	#zT = np.array([[],[]])

	#z0 = np.array([[-1.5],[-1.5]])
	#zT = np.array([[1],[1]])

	z0 = np.zeros((config.z_dim,1))
	z0[0] = 8
	z0[1] = 0

	zT = np.zeros((config.z_dim,1))
	zT[0] = -8
	zT[1] = 0

	j = 0
	for key in config.interpol_types.keys():
		for i in range(config.interpol_types[key]):

			if key == 'linear':
				path,z0,zT = linear_interpol(config,z0,zT)
			elif key == 'stoch':
				path,z0,zT = stochastic_interpol(G,D,GAN,config,z0,zT)
			elif key == 'stochSMC':
				path,z0,zT = stochasticSMC_interpol(G,D,GAN,config,z0,zT)

			paths[j,:,:] = path
			#print(path.shape)
			path_img = G.predict(path.T)
			#print(path_img.shape)
			scores = D.predict(path_img)
			#print(scores.shape)
			print(key)
			print(scores)
			print(np.mean(scores))
			print('')
			j += 1

	

	vis_interpolation(config,G,paths)

	if config.z_dim == 2:
		latent_inter(config, paths, G)


if config.metrics:

	metrics(G,D,GAN,config)

if config.heat_map:
	
	heat_map(GAN, config)

if config.latent_viz:

	if config.z_dim == 2:
		latent_visualization(config,G)
	else:

		latent_visualization_64(config,G)


"""
z = np.zeros((config.z_dim,1))
z[0] = 0
z[1] = -0

#z = get_valid_code(GAN, config)

#z = (z/np.linalg.norm(z))*8


x = G.predict(z.T)

score = D.predict(x)

print(score)


plt.figure(figsize=(1, 1))
plt.axis('off')

plt.imshow(x[0,:,:,0], cmap='gray')

filepath = '/Users/erikpiscator/Repositories/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210506-1630_lines/test'

plt.savefig(filepath)
plt.close()

"""






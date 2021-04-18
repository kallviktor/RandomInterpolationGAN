"""This is the main file used to run the program. 
Change values in the model_config input in order to change setup."""

import model
from interpolation import stochasticSMC_interpol, linear_interpol, stochastic_interpol
from interpolations_help_fcns import vis_interpolation, heat_map, latent_visualization, latent_inter, NewjointCov
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
					  metrics_k=3,
					  metrics_type='stochSMC',
					  interpol_types={'stochSMC':1,'stoch':1},
					  thresh=0.1,
					  epochs=30,
					  batch_size=64,
					  lines_batches=200,
					  z_dim=2,
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
					  progress_freq=100,
					  vis_freq=100,
					  plottrain_freq=100,
					  random_sample=False,
					  concatenate=True,
					  hm_xmin=-2,
					  hm_xmax=2,
					  hm_ymin=-2,
					  hm_ymax=2,
					  hm_steps=51,
					  out_dir='/out',
					  load_dir=r'/Users/erikpiscator/Repositories/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210224-1456_lines/models/models_2ep')


if config.loadmodel:

	print_loading_initialized()
	D = load_model(config,'D_')
	G = load_model(config,'G_')
	GAN = load_model(config,'GAN_')
	print_loading_complete()

else:

	print_training_setup(config)
	dcgan = model.dcgan(config)
	dcgan.train(config)


if config.interpolation:

	nmr_interpols = sum(config.interpol_types.values())
	paths = np.zeros((nmr_interpols,config.z_dim,config.int_steps))
	z0 = np.array([[-1],[-1.5]])
	zT = np.array([[1],[1.5]])

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
			j += 1

	vis_interpolation(config,G,paths)

	if config.z_dim == 2:
		latent_inter(config, paths, G)
		#pass


if config.metrics:

	metrics(G,D,GAN,config)

if config.heat_map:
	
	heat_map(GAN, config)

if config.latent_viz:

	latent_visualization(config,G)






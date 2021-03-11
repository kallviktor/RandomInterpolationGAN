"""This is the main file used to run the program. 
Change values in the model_config input in order to change setup."""

import model
from interpolationSMC import InterpolStochSMC, linear_interpol
from interpolations_help_fcns import vis_interpolation, heat_map, latent_visualization, latent_inter
from setup import model_config
from model_help_fcns import *
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
					  thresh=0.1,
					  epochs=30,
					  batch_size=64,
					  lines_batches=200,
					  z_dim=50,
					  z_start=0,
					  z_end=1,
					  int_time=1,
					  int_steps=10,
					  nmrParts=500,
					  gf_dim=8,
					  gfc_dim=128,
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
					  load_dir=r'/Users/erikpiscator/Documents/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210311-1553_lines/models/models_6ep')


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

	stoch_path = InterpolStochSMC(G,D,GAN,config)
	#lin_path = linear_interpol(config)

	vis_interpolation(config,G,stoch_path)
	#vis_interpolation(config,G,lin_path)

	#latent_inter(config, stoch_path, G)
	#latent_inter(config, lin_path, G)

#heat_map(GAN, config)
#latent_visualization(config,G)






"""This is the main file used to run the program. 
Change values in the model_config input in order to change setup."""

import model
from interpolationSMC import InterpolStochSMC
from setup import model_config
from model_help_fcns import save_model, load_model


#create configuration object
config = model_config(dataset='mnist',
					  loadmodel=False,
					  interpolation=False,
					  epochs=2,
					  batch_size=64,
					  lines_batches=1000,
					  z_dim=100,
					  z_start=0,
					  z_end=1,
					  int_time=1,
					  int_steps=10,
					  nmrParts=100,
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
					  out_dir='/out',
					  load_dir='/Users/erikpiscator/Documents/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210224-1456_lines/models/models_2ep')


if config.loadmodel:

	D = load_model(config,'D_')
	G = load_model(config,'G_')
	GAN = load_model(config,'GAN_')

else:

	dcgan = model.dcgan(config)
	dcgan.train(config)

if config.interpolation:

	interpol = InterpolStochSMC(G,D,GAN,config)
	print(interpol.shape)








"""This is the main file used to run the program. 
Change values in the model_config input in order to change setup."""

import model
#import interpolations
from setup import model_config
import keras
from utils import save_model, load_model


#create configuration object
config = model_config(dataset='lines',loadmodel=True,interpolation=False,
						epochs=10,batch_size=64,
						z_dim=2,gf_dim=32,df_dim=32,c_dim=1,
						progress_freq=100,vis_freq=100,plottrain_freq=100,
						learning_rate= 0.0002,clip=0.01,n_critic=5,
						optimizer='Adam', loss_f='Goodfellow', random_sample=False,
						concatenate=True,lines_batches=100,
						load_dir='/Users/erikpiscator/Documents/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210224-1456_lines/models/models_1ep')


if config.loadmodel:

	D = load_model(config,'D_')
	G = load_model(config,'G_')
	GAN = load_model(config,'GAN_')

	D.summary()
	

else:
	dcgan = model.dcgan(config)
	dcgan.train(config)


if config.interpolation:
	pass
	#interpolations.interpolation_stochastic()








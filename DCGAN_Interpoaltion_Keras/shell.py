import model
#import interpolations
from setup import model_config
import keras
from utils import save_model, load_model


#create configuration object
config = model_config(dataset='mnist',loadmodel=False,interpolation=False,
						epochs=3,batch_size=64,
						z_dim=100,gf_dim=32,df_dim=32,c_dim=1,
						progress_freq=10,vis_freq=100,plottrain_freq=500,
						learning_rate=0.0002,
						load_dir='/Users/erikpiscator/Documents/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210218-1143_mnist/models')


if config.loadmodel:

	D = load_model(config,'D_')
	G = load_model(config,'G_')
	GAN = load_model(config,'GAN_')
	 
else:
	dcgan = model.dcgan(config)
	#dcgan.D.summary()
	#dcgan.G.summary()
	
	dcgan.train(config)
	save_model(config,dcgan)

if config.interpolation:
	pass
	#interpolations.interpolation_stochastic()








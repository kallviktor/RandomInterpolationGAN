import model
#import interpolations
from setup import model_config
import keras
from utils import save_model, load_model


#create configuration object
config = model_config(dataset='lines',loadmodel=False,interpolation=False,
						epochs=10,batch_size=64,
						z_dim=2,gf_dim=32,df_dim=32,c_dim=1,
						progress_freq=100,vis_freq=100,plottrain_freq=100,
						learning_rate= 0.0002,clip=0.01,n_critic=5,
						optimizer='Adam', loss_f='Goodfellow', random_sample=False,
						concatenate=True,lines_batches=100,
						load_dir='/Users/erikpiscator/Documents/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210219-2232_mnist/models')


if config.loadmodel:

	#problem with loading models when custom loss function is used
	#ValueError: Unknown loss function: G_lossfunc
	D = load_model(config,'D_')
	G = load_model(config,'G_')
	GAN = load_model(config,'GAN_')
	

else:
	dcgan = model.dcgan(config)
	dcgan.train(config)


if config.interpolation:
	pass
	#interpolations.interpolation_stochastic()








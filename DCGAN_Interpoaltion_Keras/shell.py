import model
from setup import model_config
import keras
from utils import save_model, load_model


#create configuration object
config = model_config(dataset='mnist',loadmodel=True,interpolation=False,epochs=1,batch_size=64,
						z_dim=100,gf_dim=64,df_dim=64,gfc_dim=1024,dfc_dim=1024,
						c_dim=1,progress_freq=10,load_dir='/Users/erikpiscator/Documents/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210218-1131_mnist/models')


if config.loadmodel:

	D = load_model(config,'D_')
	G = load_model(config,'G_')
	GAN = load_model(config,'GAN_')
	 
else:
	dcgan = model.dcgan(config)
	dcgan.train(config)
	save_model(config,dcgan)

if config.interpolation:
	
	pass

G.summary()







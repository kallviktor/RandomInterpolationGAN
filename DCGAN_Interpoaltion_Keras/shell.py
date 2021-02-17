import model
from setup import model_config
import keras


#create configuration object
config = model_config(dataset='mnist',loadmodel=False,interpolation=False,epochs=1,batch_size=64,
						z_dim=100,gf_dim=64,df_dim=64,gfc_dim=1024,dfc_dim=1024,
						c_dim=1,progress_freq=10,load_dir='/Users/erikpiscator/Documents/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210217-2101_mnist/mnist_100d_1ep.h5')


if config.loadmodel:
	dcgan = keras.models.load_model(config.load_dir)
	pass 
else:
	dcgan = model.dcgan(config)
	dcgan.train(config)
	dcgan.save_model(config)


if config.interpolation:
	
	pass









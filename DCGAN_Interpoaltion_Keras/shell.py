import model
#import interpolations
from setup import model_config
import keras
from utils import save_model, load_model


#create configuration object
config = model_config(dataset='mnist',loadmodel=False,interpolation=False,
						epochs=10,batch_size=64,
						z_dim=100,gf_dim=32,df_dim=32,c_dim=1,
						progress_freq=100,vis_freq=500,plottrain_freq=100,
						learning_rate=0.0002,
						load_dir='/Users/erikpiscator/Documents/RandomInterpolationGAN/DCGAN_Interpoaltion_Keras/out/20210219-2232_mnist/models')


if config.loadmodel:

	#problem with loading models when custom loss function is used
	#ValueError: Unknown loss function: G_lossfunc
	D = load_model(config,'D_')
	G = load_model(config,'G_')
	GAN = load_model(config,'GAN_')
	
	batch_z = np.random.normal(0,1,size=(1,config.z_dim))
	prediction = G.predict(batch_z)
	prediction = prediction.reshape((config.x_h, config.x_w))
	prediction = prediction*127.5 + 127.5
	plt.imshow(prediction,cmap='gray')
	plt.savefig(config.images_dir+'/vis_testing')

else:
	dcgan = model.dcgan(config)
	#dcgan.D.summary()
	#dcgan.G.summary()
	
	dcgan.train(config)
	save_model(config,dcgan)

if config.interpolation:
	pass
	#interpolations.interpolation_stochastic()








import model
from setup import model_config


#create configuration object
config = model_config(dataset='mnist',loadmodel=False,interpolation=False,epochs=1,batch_size=64,
						z_dim=100,gf_dim=64,df_dim=64,gfc_dim=1024,dfc_dim=1024,
						c_dim=1,sample_freq=10)


if config.loadmodel:
	#load existing model
	pass 
else:
	dcgan = model.dcgan(config)
	dcgan.train(config)

if config.interpolation:
	#do interpolation
	pass









import keras.backend as K
from keras.datasets import mnist
from keras.models import load_model as load
from keras.metrics import Mean
import math
import numpy as np
import time
import datetime
import os
import glob
import matplotlib.pyplot as plt

def load_mnist():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	X_train = np.pad(X_train, ((0,0),(2,2),(2,2)), 'constant')
	X_test = np.pad(X_test, ((0,0),(2,2),(2,2)), 'constant')

	return (X_train, y_train), (X_test, y_test)


def load_lines():
	pass


def load_celebA():
	pass


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def print_training_progress(config,epoch,batch,batches,D_loss,G_loss,start_time,t0):

	dt = round(time.time()-t0,1)
	ET = datetime.timedelta(seconds=round(time.time()-start_time,0))
	

	if epoch == 0 and batch+1 == config.progress_freq:
		est_time = datetime.timedelta(seconds=round(dt*batches*config.epochs/config.progress_freq,0))
		print('Estimated time to completion: {}'.format(est_time))
		print('\n'*1)
	
	print('Epoch: {}/{} | Batch: {}/{} | ET: {} | dt: {}s | D-loss: {:1.2e} | G-loss: {:1.2e}'.format(epoch+1,config.epochs,batch+1,batches,ET,dt,D_loss,G_loss))

def print_training_initialized():
	print('\n' * 1)
	print('='*65)
	print('-'*21,'Training initialized','-'*22)
	print('='*65)
	print('\n' * 1)

def print_training_complete():
	print('\n' * 1)
	print('='*65)
	print('-'*23,'Training complete','-'*23)
	print('='*65)

def save_model(config,model):

	if not os.path.exists(config.out_dir):
			os.makedirs(config.out_dir)

	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	if not os.path.exists(config.models_dir):
		os.makedirs(config.models_dir)

	model.D.save(config.models_dir+'/D_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,config.epochs))
	model.G.save(config.models_dir+'/G_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,config.epochs))
	model.GAN.save(config.models_dir+'/GAN_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,config.epochs))

def load_model(config,model_type):

	file_path = glob.glob(config.load_dir+'/{}*.h5'.format(model_type))
	model = load(file_path[0], compile = True)

	return model

def save_gen_imgs(config,G,epoch,batch):

	image_frame_dim = int(math.ceil(config.batch_size**.5))

	#batch_z = np.random.normal(0,1,size=(config.batch_size,config.z_dim))
	batch_z = np.random.normal(0,1,size=(1,config.z_dim))
	prediction = G.predict(batch_z)
	prediction = prediction.reshape((config.x_h, config.x_w))
	prediction = prediction*127.5 + 127.5
	plt.imshow(prediction,cmap='gray')

	if not os.path.exists(config.out_dir):
			os.makedirs(config.out_dir)

	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	if not os.path.exists(config.images_dir):
		os.makedirs(config.images_dir)

	plt.savefig(config.images_dir+'/vis_{}ep_{}batch'.format(epoch+1,batch+1))

def G_lossfunc(y_true,y_pred):
	
	#return K.mean(K.log(1-y_pred))
	return K.mean(-K.log(y_pred))

def D_lossfunc(y_true,y_pred):
	
	y_pred_real = y_pred[0:int(len(y_true)/2)]
	y_pred_fake = y_pred[int(len(y_true)/2):]

	loss = K.mean(-K.log(y_pred_real)-K.log(1-y_pred_fake))

	return loss





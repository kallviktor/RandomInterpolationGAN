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
from numpy.random import randint

def load_mnist():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	#X_train = np.pad(X_train, ((0,0),(2,2),(2,2)), 'constant')
	X_train = (X_train.astype(np.float32) - 127.5)/127.5
	#X_test = np.pad(X_test, ((0,0),(2,2),(2,2)), 'constant')
	X_test = (X_test.astype(np.float32) - 127.5)/127.5

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
	
	print('Epoch: {}/{} | Batch: {}/{} | ET: {} | dt: {}s | D-loss: {:1.2e} | G-loss: {:1.2e}'.format(epoch+1,config.epochs,batch,batches,ET,dt,D_loss,G_loss))

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

def save_model_checkpoint(config,epoch,D,G,GAN):

	if not os.path.exists(config.out_dir):
		os.makedirs(config.out_dir)

	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	if not os.path.exists(config.models_dir):
		os.makedirs(config.models_dir)

	D.save(config.models_dir+'/D_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,epoch+1))
	G.save(config.models_dir+'/G_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,epoch+1))
	GAN.save(config.models_dir+'/GAN_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,epoch+1))


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
	plt.axis('off')
	plt.imshow(prediction,cmap='gray')

	if not os.path.exists(config.out_dir):
			os.makedirs(config.out_dir)

	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	if not os.path.exists(config.images_dir):
		os.makedirs(config.images_dir)

	plt.savefig(config.images_dir+'/vis_{}ep_{}batch'.format(epoch+1,batch+1))
	plt.close()

def save_gen_imgs_new(config,G,epoch,batch):

	image_frame_dim = int(math.ceil(config.batch_size**.5))
	#fig, axs = plt.subplots(image_frame_dim, image_frame_dim)
	fig = plt.figure(figsize=(image_frame_dim,image_frame_dim)) # Notice the equal aspect ratio
	axs = [fig.add_subplot(image_frame_dim,image_frame_dim,i+1) for i in range(image_frame_dim*image_frame_dim)]

	#batch_z = np.random.normal(0,1,size=(config.batch_size,config.z_dim))
	
	counter = 0
	#for i in range(image_frame_dim):
	#	for j in range(image_frame_dim):
	for ax in axs:
			batch_z = np.random.normal(0,1,size=(1,config.z_dim))
			prediction = G.predict(batch_z)
			prediction = prediction.reshape((config.x_h, config.x_w))
			prediction = prediction*127.5 + 127.5
			ax.imshow(prediction,cmap='gray')
			ax.axis('off')
			ax.set_aspect('equal')
			counter += 1

	if not os.path.exists(config.out_dir):
			os.makedirs(config.out_dir)

	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	if not os.path.exists(config.images_dir):
		os.makedirs(config.images_dir)

	plt.subplots_adjust(wspace=0.015, hspace=0.015)
	plt.savefig(config.images_dir+'/vis_{}ep_{}batch'.format(epoch+1,batch))
	plt.close()
	

def plot_save_train_prog(config,D_loss_vec,G_loss_vec,batch_vec,epoch,batch):

	if not os.path.exists(config.out_dir):
			os.makedirs(config.out_dir)

	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	if not os.path.exists(config.images_dir):
		os.makedirs(config.images_dir)


	plt.plot(batch_vec,D_loss_vec,label='D-loss')
	plt.plot(batch_vec,G_loss_vec,label='G-loss')

	plt.xlabel('Batch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(config.images_dir+'/trainprog_{}ep_{}batch'.format(epoch+1,batch))
	plt.close()

def G_lossfunc(y_true,y_pred):
	
	#return K.mean(K.log(1-y_pred))
	return K.mean(-K.log(y_pred))

def D_lossfunc(y_true,y_pred):
	
	y_pred_real = y_pred[0:int(len(y_true)/2)]
	y_pred_fake = y_pred[int(len(y_true)/2):]

	loss = K.mean(-K.log(y_pred_real)-K.log(1-y_pred_fake))

	return loss

def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)

def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	
	return X

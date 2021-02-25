import keras.backend as K
from keras.datasets import mnist
from keras.models import load_model as load
from keras.metrics import Mean
from keras.optimizers import SGD, Adam, RMSprop
from keras.initializers import RandomNormal
import math
import numpy as np
import time
import datetime
import os
import glob
import matplotlib.pyplot as plt
from numpy.random import randint
from lines import *

def load_mnist():
	"""Loads the 'mnist' dataset. Scaling is applied to get data in range [-1,1].
	   Adding chanel dimension with expand_dims.
	   Upscaling images from 28x28 to 32x32 for easier transition between datasets.
	   Most datasets has a reslotion that is a multiple of 2."""

	(X_train, _), (_, _) = mnist.load_data()

	X_train = np.pad(X_train, ((0,0),(2,2),(2,2)), 'constant')
	X_train = (X_train.astype(np.float32) - 127.5)/127.5
	X_train = np.expand_dims(X_train,axis=-1)

	return X_train


def load_lines(batch_size):
	"""Loads a batch of the 'lines' dataset. The dataset is generated on the fly
	   with the get_dataset from the lines.py file."""

	dataset = get_dataset('lines32',dict(batch_size = batch_size))
	X_train = next(iter(dataset.train))['x']

	return X_train

def load_celebA():
	"""Loads the celebA dataset."""
	pass


def conv_out_size_same(size, stride):
	"""Function not in use atm."""
	return int(math.ceil(float(size) / float(stride)))

def print_training_setup(config):
	"""Printing the setup of the program defined in the config object from the
	   model_config class in setup.py."""

	print('\n'*1)
	print('-'*24,'Training setup','-'*25)
	print('Dataset: {}'.format(config.dataset))
	print('Epochs: {}'.format(config.epochs))
	print('Batch size: {}'.format(config.batch_size))
	print('Latent dim: {}'.format(config.z_dim))
	print('Optimizer: {}'.format(config.optimizer))
	print('Loss function: {}'.format(config.loss_f))
	print('Learning rate: {}'.format(config.learning_rate))
	print('-'*65)

def print_training_progress(config,epoch,batch,batches,D_loss,G_loss,start_time,t0):
	"""Printing the progress of the training."""

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
	"""Function not in use atm."""
	if not os.path.exists(config.out_dir):
		os.makedirs(config.out_dir)

	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	if not os.path.exists(config.models_dir):
		os.makedirs(config.models_dir)

	model.D.save(config.models_dir+'/D_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,config.epochs))
	model.G.save(config.models_dir+'/G_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,config.epochs))
	model.GAN.save(config.models_dir+'/GAN_{}_{}d_{}ep.h5'.format(config.dataset,config.z_dim,config.epochs))

def save_model_checkpoint(config,epoch,batches,D,G,GAN):
	"""Saves models during and after training. Creates directories if non existing."""

	checkpoint_dir = config.models_dir+'/models_{}ep'.format(epoch+1)

	if not os.path.exists(config.out_dir):
		os.makedirs(config.out_dir)

	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	if not os.path.exists(config.models_dir):
		os.makedirs(config.models_dir)

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	D.save(checkpoint_dir+'/D_{}_{}d_{}ep_{}b_{}bs.h5'.format(config.dataset,config.z_dim,epoch+1,batches,config.batch_size))
	G.save(checkpoint_dir+'/G_{}_{}d_{}ep_{}b_{}bs.h5'.format(config.dataset,config.z_dim,epoch+1,batches,config.batch_size))
	GAN.save(checkpoint_dir+'/GAN_{}_{}d_{}ep_{}b_{}bs.h5'.format(config.dataset,config.z_dim,epoch+1,batches,config.batch_size))


def load_model(config,model_type):
	"""Loads models from config.load_dir. Currently no loading trainable model.
	   Set compile=True and give the correct loss function in custom_objects
	   in order to continue training."""

	file_path = glob.glob(config.load_dir+'/{}*.h5'.format(model_type))

	if model_type == 'D_':
		model = load(file_path[0], compile = False)#, custom_objects={'goodfellow_loss_D': goodfellow_loss_D})
	else:
		model = load(file_path[0], compile = False)#, custom_objects={'goodfellow_loss_G': goodfellow_loss_G})

	return model

def save_gen_imgs(config,G,epoch,batch):
	"""Plots and saves one generated image from G."""
	image_frame_dim = int(math.ceil(config.batch_size**.5))

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

def save_gen_imgs_batch(config,G,epoch,batch):
	"""Plots and saves (batch_size x batch_size) generated images from G."""

	image_frame_dim = int(math.ceil(config.batch_size**.5))
	fig = plt.figure(figsize=(image_frame_dim,image_frame_dim)) # Notice the equal aspect ratio
	axs = [fig.add_subplot(image_frame_dim,image_frame_dim,i+1) for i in range(image_frame_dim*image_frame_dim)]
	
	counter = 0

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
	plt.savefig(config.images_dir+'/vis_{}ep_{}batch'.format(epoch+1,batch+1))
	plt.close()
	

def plot_save_train_prog(config,D_loss_vec,G_loss_vec,batch_vec,epoch,batch):
	"""Saves plots of the D and G loss aginst number of batches. Creates directories if non existing."""
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
	plt.savefig(config.images_dir+'/trainprog_{}ep_{}batch'.format(epoch+1,batch+1))
	plt.close()

def goodfellow_loss_G(y_true,y_pred):
	"""Slightly modified G-loss function from original GAN paper."""

	#return K.mean(K.log(1-y_pred))
	return K.mean(-K.log(y_pred))

def goodfellow_loss_D(y_true,y_pred):
	"""D-loss function from original GAN paper."""

	m = len(y_pred)

	y_pred_real = y_pred[0:int(len(y_true)/2)]
	y_pred_fake = y_pred[int(len(y_true)/2):]

	return K.sum(-K.log(y_pred_real)-K.log(1-y_pred_fake)) / m

def wasserstein_loss(y_true, y_pred):
	"""Lossfunction from WGAN"""

	return K.mean(y_true * y_pred)

def generate_real_samples(config,dataset,random=False,batch=None):
	"""Generates a batch from a given dataset, either ransom or deterministic."""

	n_samples = int(config.batch_size/2)

	if random:
		ix = randint(0, dataset.shape[0], n_samples)
		X = dataset[ix]
		y = np.zeros((int(n_samples)))
	else:
		X = dataset[int(batch*n_samples):int((batch+1)*n_samples)]
		y = np.zeros((int(n_samples)))

	return X, y

def generate_real_samples_lines(config):
	"""Generates a batch from 'lines' dataset."""
	
	batch_size = int(config.batch_size/2)
	X = load_lines(batch_size)
	y = np.zeros((int(batch_size)))

	return X, y

def generate_fake_samples(config,G):
	"""Generates a batch of fake samples from G."""

	n_samples = int(config.batch_size/2)

	batch_z = np.random.normal(0,1,size=(n_samples,config.z_dim))
	X = G.predict(batch_z)
	y = np.ones((int(n_samples)))

	return X, y

def generate_latent_codes(config):
	"""Generates a batch of codes from latent space."""

	y = np.zeros((config.batch_size))
	z = np.random.normal(0,1,size=(int(config.batch_size),config.z_dim))

	return z, y

def get_loss_opt(config,model):
	"""Sets the loss function and optimizer according to the setup."""

	if config.optimizer == 'Adam':
		opt = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1)
	elif config.optimizer == 'RMSprop':
		opt = RMSprop(lr=config.learning_rate)
	else:
		opt = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1)

	if config.loss_f == 'Goodfellow':
		if model == 'GAN':
			loss = goodfellow_loss_G
		elif model == 'D':
			loss = goodfellow_loss_D
	elif config.loss_f == 'Wasserstein':
		loss = wasserstein_loss
	elif config.loss_f == 'binary_crossentropy':
		loss = 'binary_crossentropy'
	else:
		loss = 'binary_crossentropy'

	return loss, opt

def get_init(config):
	"""Sets the kernel initializer for the networks."""

	if config.init == 'RandomNormal':
		init = RandomNormal(stddev=config.init_stddev)

	return init

def train_D(config,D,X_real,y_real,X_fake,y_fake):
	"""Training D with either a concatinated batch of real and fake samples
	   or two separate batches."""

	if config.concatenate:
		X = np.concatenate((X_real,X_fake),axis=0)
		y = np.concatenate((y_real,y_fake),axis=0)

		D_loss = D.train_on_batch(X,y)
	else:
		D_loss_real = D.train_on_batch(X_real,y_real)
		D_loss_fake = D.train_on_batch(X_fake,y_fake)

		D_loss = 0.5*(D_loss_real+D_loss_fake)

	return D_loss

def train_G(config,GAN,z,y):
	"""Training G."""
	G_loss = GAN.train_on_batch(z,y)

	return G_loss


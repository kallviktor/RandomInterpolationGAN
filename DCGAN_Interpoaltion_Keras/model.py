from keras.models import Sequential
from keras.layers import Dense, Reshape, ReLU, LeakyReLU, BatchNormalization as BN#, tanh, sigmoid
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist
from keras.initializers import RandomNormal
from keras.constraints import Constraint
import keras.backend as K

import numpy as np
from numpy import expand_dims
import math
import time
import datetime
import os

from utils import *


class dcgan(object):

	def __init__(self, config):
		"""
		Args:
		batch_size: The size of batch. Should be specified before training.
		y_dim: (optional) Dimension of dim for y. [None]
		z_dim: (optional) Dimension of dim for Z. [100]
		gf_dim: (optional) Dimension of G filters in first conv layer. [64]
		df_dim: (optional) Dimension of D filters in first conv layer. [64]
		gfc_dim: (optional) Dimension of G units for for fully connected layer. [1024]
		dfc_dim: (optional) Dimension of D units for fully connected layer. [1024]
		c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
		"""

		self.build_model(config)

	def build_model(self,config):
		"""
		self.D = self.discriminator(config)
		self.G = self.generator(config)

		self.GAN = Sequential()
		self.GAN.add(self.G)
		self.GAN.add(self.D)
		"""
		self.D = self.discriminator(config)
		self.G = self.generator(config)
		# make weights in the critic not trainable
		for layer in self.D.layers:
			if not isinstance(layer, BN):
				layer.trainable = False
		# connect them
		self.GAN = Sequential()
		# add generator
		self.GAN.add(self.G)
		# add the critic
		self.GAN.add(self.D)
		# compile model
		opt = RMSprop(lr=0.00005)
		self.GAN.compile(loss=wasserstein_loss, optimizer=opt)
		


	def discriminator(self,config):

		init = RandomNormal(stddev=0.02)
		const = ClipConstraint(config.clip)
		input_shape = (config.x_h,config.x_w,config.x_d)
		"""
		D = Sequential()

		D.add(Conv2D(filters=config.df_dim,strides=2,padding='same',kernel_size=5,kernel_initializer=init,kernel_constraint=const,input_shape=input_shape))
		D.add(LeakyReLU(alpha=0.2))

		D.add(Conv2D(filters=config.df_dim*2,strides=2,padding='same',kernel_size=5,kernel_constraint=const,kernel_initializer=init))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))
		

		D.add(Conv2D(filters=config.df_dim*4,strides=2,padding='same',kernel_size=5,kernel_constraint=const,kernel_initializer=init))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))

		#if config.dataset not in ['mnist','lines']:
		D.add(Conv2D(filters=config.df_dim*8,strides=2,padding='same',kernel_size=5,kernel_constraint=const,kernel_initializer=init))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))

		D.add(Flatten())
		D.add(Dense(1))
		#D.add(Activation('sigmoid'))
		"""
			# define model
		model = Sequential()
		# downsample to 14x14
		model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=input_shape))
		model.add(BN())
		model.add(LeakyReLU(alpha=0.2))
		# downsample to 7x7
		model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
		model.add(BN())
		model.add(LeakyReLU(alpha=0.2))
		# scoring, linear activation
		model.add(Flatten())
		model.add(Dense(1))
		# compile model
		opt = RMSprop(lr=0.00005)
		model.compile(loss=wasserstein_loss, optimizer=opt)

		#print('D:')
		#D.summary()

		return model

	def generator(self,config):

		init = RandomNormal(stddev=0.02)
		"""
		G = Sequential()

		G.add(Dense(input_dim=config.z_dim,kernel_initializer=init,units=config.gf_dim*8*4*4))
		G.add(Reshape((4,4,config.gf_dim*8)))
		G.add(BN(momentum=0.9,epsilon=1e-5))
		G.add(ReLU())

		G.add(Conv2DTranspose(filters=config.gf_dim*4,strides=2,padding='same',kernel_size=5,kernel_initializer=init))
		G.add(BN(momentum=0.9,epsilon=1e-5))
		G.add(ReLU())

		G.add(Conv2DTranspose(filters=config.gf_dim*2,strides=2,padding='same',kernel_size=5,kernel_initializer=init))
		G.add(BN(momentum=0.9,epsilon=1e-5))
		G.add(ReLU())

		if config.dataset not in ['mnist','lines']:
			#more layers could (and should) be added in order to get correct output size of G

			G.add(Conv2DTranspose(filters=config.gf_dim,strides=2,padding='same',kernel_size=5,kernel_initializer=init))
			G.add(BN(momentum=0.9,epsilon=1e-5))
			G.add(ReLU())

		G.add(Conv2DTranspose(filters=config.c_dim,strides=2,padding='same',kernel_size=5,kernel_initializer=init))
		G.add(Activation('tanh'))
		"""
		#print('G:')
		#G.summary()

		model = Sequential()
		# foundation for 7x7 image
		n_nodes = 128 * 7 * 7
		model.add(Dense(n_nodes, kernel_initializer=init, input_dim=config.z_dim))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Reshape((7, 7, 128)))
		# upsample to 14x14
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(BN())
		model.add(LeakyReLU(alpha=0.2))
		# upsample to 28x28
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(BN())
		model.add(LeakyReLU(alpha=0.2))
		# output 28x28x1
		model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))

		return model


	def train(self,config):

		if config.dataset == 'mnist':
			(X_train, y_train), (X_test, y_test) = load_mnist()
			X_train = expand_dims(X_train,axis=-1)
		elif config.dataset == 'lines':
			(X_train, y_train), (X_test, y_test) = load_lines()
		elif config.dataset == 'celebA':
			(X_train, y_train), (X_test, y_test) = load_celebA()

		#D_optim = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1)
		D_optim = RMSprop(learning_rate=config.learning_rate)
		#G_optim = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1)
		G_optim = RMSprop(learning_rate=config.learning_rate)
		loss_f = 'binary_crossentropy'


		#Compile models
		self.D.compile(loss=wasserstein_loss,optimizer=D_optim)
		self.D.trainable = False
		self.G.compile(loss=G_lossfunc,optimizer=G_optim)
		self.GAN.compile(loss=wasserstein_loss,optimizer=G_optim)

		batches = int(len(X_train)/config.batch_size)		#int always rounds down --> no problem with running out of data
		
		D_loss_vec = []
		G_loss_vec = []
		batch_vec = []

		D_loss_vec.append(0)
		G_loss_vec.append(0)
		batch_vec.append(0)

		counter = 0

		print_training_initialized()

		start_time = time.time()
		t0 = start_time
		for epoch in range(config.epochs):
			for batch in range(0,batches,config.n_critic):

				#Update D network more times than G, according to Wasserstein
				for i in range(config.n_critic):

					#batch_X_real = X_train[int((batch+i)*config.batch_size/2):int((batch+i+1)*config.batch_size/2)][np.newaxis].transpose(1,2,3,0)
					#batch_X_real = generate_real_samples(X_train,int(config.batch_size/2))[np.newaxis].transpose(1,2,3,0)
					batch_X_real = generate_real_samples(X_train,int(config.batch_size/2))
					batch_z = np.random.normal(0,1,size=(int(config.batch_size/2),config.z_dim))
					batch_X_fake = self.G.predict(batch_z)
					#batch_X = np.concatenate((batch_X_real,batch_X_fake),axis=0)

					#batch_yd = np.concatenate((np.ones((int(config.batch_size/2))),-np.ones((int(config.batch_size/2)))))
					batch_yd_real = -np.ones((int(config.batch_size/2)))
					batch_yd_fake = np.ones((int(config.batch_size/2)))

					D_loss_real = self.D.train_on_batch(batch_X_real,batch_yd_real)
					D_loss_fake = self.D.train_on_batch(batch_X_fake,batch_yd_fake)

					D_loss = 0.5*(D_loss_real+D_loss_fake)

				#Update G network
				
				batch_yg = -np.ones((config.batch_size))
				batch_z = np.random.normal(0,1,size=(int(config.batch_size),config.z_dim))
				G_loss = self.GAN.train_on_batch(batch_z, batch_yg)

				#Update G network again according to https://github.com/carpedm20/DCGAN-tensorflow.git
				#batch_z = np.random.normal(0,1,size=(int(config.batch_size),config.z_dim))
				#G_loss = self.GAN.train_on_batch(batch_z, batch_yg)

				#Save losses to vectors in order to plot
				D_loss_vec.append(D_loss)
				G_loss_vec.append(G_loss)
				batch_vec.append(counter)

				#save generated images
				if np.mod(counter,config.vis_freq) == 0 or (epoch==0 and batch==0):
					#save_gen_imgs_new(config,self.G,epoch,batch)
					save_gen_imgs_new(config,self.G,epoch,batch)

				#plot training progress
				if np.mod(counter,config.plottrain_freq) == 0 and counter != 0:
					plot_save_train_prog(config,D_loss_vec,G_loss_vec,batch_vec,epoch,batch)

				#Print status and save images for each config.sample_freq iterations
				if np.mod(counter,config.progress_freq) == 0 and counter != 0:
					print_training_progress(config,epoch,batch,batches,D_loss,G_loss,start_time,t0)
					t0 = time.time()

				counter += config.n_critic
			#save model after each epoch
			save_model_checkpoint(config,epoch,self.D,self.G,self.GAN)

		print_training_complete()


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return K.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}







from keras.models import Sequential
from keras.layers import Dense, Reshape, ReLU, LeakyReLU, BatchNormalization as BN#, tanh, sigmoid
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.initializers import RandomNormal


import numpy as np
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

		self.D = self.discriminator(config)
		self.G = self.generator(config)

		self.GAN = Sequential()
		self.GAN.add(self.G)
		self.D.trainable = False
		self.GAN.add(self.D)


	def discriminator(self,config):

		init = RandomNormal(stddev=0.02)
		input_shape = (config.x_h,config.x_w,config.x_d)

		D = Sequential()

		D.add(Conv2D(filters=config.df_dim,strides=2,padding='same',kernel_size=5,kernel_initializer=init,input_shape=input_shape))
		D.add(LeakyReLU(alpha=0.2))

		D.add(Conv2D(filters=config.df_dim*2,strides=2,padding='same',kernel_size=5,kernel_initializer=init))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))

		

		D.add(Conv2D(filters=config.df_dim*4,strides=2,padding='same',kernel_size=5,kernel_initializer=init))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))

		#if config.dataset not in ['mnist','lines']:
		D.add(Conv2D(filters=config.df_dim*8,strides=2,padding='same',kernel_size=5,kernel_initializer=init))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))

		D.add(Flatten())
		D.add(Dense(1))
		D.add(Activation('sigmoid'))

		#print('D:')
		#D.summary()

		return D

	def generator(self,config):

		init = RandomNormal(stddev=0.02)

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

		#print('G:')
		#G.summary()

		return G


	def train(self,config):

		if config.dataset == 'mnist':
			(X_train, y_train), (X_test, y_test) = load_mnist()
			X_train = (X_train.astype(np.float32) - 127.5)/127.5
		elif config.dataset == 'lines':
			(X_train, y_train), (X_test, y_test) = load_lines()
		elif config.dataset == 'celebA':
			(X_train, y_train), (X_test, y_test) = load_celebA()

		D_optim = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1)
		G_optim = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1)
		loss_f = 'binary_crossentropy'

		#Compile models
		self.D.compile(loss=loss_f,optimizer=D_optim)
		self.D.trainable = True
		self.G.compile(loss=loss_f,optimizer=G_optim)
		self.GAN.compile(loss=loss_f,optimizer=G_optim)

		batches = int(len(X_train)/config.batch_size)		#int always rounds down --> no problem with running out of data

		counter = 1

		print_training_initialized()

		start_time = time.time()
		t0 = start_time
		for epoch in range(config.epochs):
			for batch in range(batches):

				batch_X_real = X_train[int(batch*config.batch_size/2):int((batch+1)*config.batch_size/2)][np.newaxis].transpose(1,2,3,0)

				batch_z = np.random.normal(0,1,size=(int(config.batch_size/2),config.z_dim))
				batch_X_fake = self.G.predict(batch_z)
				#batch_X = np.concatenate((batch_X_real,batch_X_fake),axis=0)

				#batch_yd = np.concatenate((np.ones((config.batch_size)),np.zeros((config.batch_size))))
				batch_yd_real = np.ones((int(config.batch_size/2)))
				batch_yd_fake = np.zeros((int(config.batch_size/2)))
				batch_yg = np.ones((config.batch_size))

				#Update D network
				self.D.trainable = True
				D_loss_real = self.D.train_on_batch(batch_X_real, batch_yd_real)
				D_loss_fake = self.D.train_on_batch(batch_X_fake, batch_yd_fake)
				D_loss = D_loss_real+D_loss_fake

				#Update G network
				batch_z = np.random.normal(0,1,size=(int(config.batch_size),config.z_dim))
				self.D.trainable = False
				G_loss = self.GAN.train_on_batch(batch_z, batch_yg)

				#Update G network again according to https://github.com/carpedm20/DCGAN-tensorflow.git
				G_loss = self.GAN.train_on_batch(batch_z, batch_yg)


				#Save losses to vectors in order to plot
				if np.mod(counter,config.vis_freq) == 0:
					save_gen_imgs(config,self.G,epoch,batch)

				if np.mod(counter,config.plottrain_freq) == 0:
					pass

				#Print status and save images for each config.sample_freq iterations
				if np.mod(counter,config.progress_freq) == 0:

					print_training_progress(config,epoch,batch,batches,D_loss,G_loss,start_time,t0)
					t0 = time.time()

				counter += 1

		print_training_complete()








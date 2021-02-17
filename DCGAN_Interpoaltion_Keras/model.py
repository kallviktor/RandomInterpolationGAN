from keras.models import Sequential
from keras.layers import Dense, Reshape, ReLU, LeakyReLU, BatchNormalization as BN
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose, tanh, sigmoid
from keras.optimizers import SGD, Adam
from keras.datasets import mnist

import numpy as np
import math

from utils import load_mnist, load_lines, load_celebA


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

		self.build_model()

	def build_model(self,config):

		self.D = self.discriminator(config)
		self.G = self.generator(config)

		self.GAN = Sequential()
		self.GAN.add(self.G)
		self.D.trainable = False
		self.GAN.add(self.D)


	def discriminator(self,config):


		input_shape = (config.batch_size,config.x_h,config.x_w,config.x_d)

		D = Sequential()

		D.add(Conv2D(filters=config.df_dim,strides=2,padding='same',kernel=5,input_shape=input_shape))
		D.add(LeakyReLU(alpha=0.2))

		D.add(Conv2D(filters=config.df_dim*2,strides=2,padding='same',kernel=5))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))

		D.add(Conv2D(filters=config.df_dim*4,strides=2,padding='same',kernel=5))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))

		D.add(Conv2D(filters=config.df_dim*8,strides=2,padding='same',kernel=5))
		D.add(BN(momentum=0.9,epsilon=1e-5))
		D.add(LeakyReLU(alpha=0.2))

		D.add(Flatten())
		D.add(Dense(1))
		D.add(sigmoid())

		return D

	def generator(self,config):

		G = Sequential()

		G.add(Dense(input_dim=config.z_dim, output_dim=config.gf_dim*8*4*4))
		G.add(Reshape(config.gf_dim*8,4,4))
		G.add(BN(momentum=0.9,epsilon=1e-5))
		G.add(ReLU())

		G.add(Conv2DTranspose(filters=config.gf_dim*4,strides=2,padding='same',kernel=5))
		G.add(BN(momentum=0.9,epsilon=1e-5))
		G.add(ReLU())

		G.add(Conv2DTranspose(filters=config.gf_dim*2,strides=2,padding='same',kernel=5))
		G.add(BN(momentum=0.9,epsilon=1e-5))
		G.add(ReLU())

		G.add(Conv2DTranspose(filters=config.gf_dim,strides=2,padding='same',kernel=5))
		G.add(BN(momentum=0.9,epsilon=1e-5))
		G.add(ReLU())

		G.add(Conv2DTranspose(filters=config.c_dim,strides=2,padding='same',kernel=5))
		G.add(tanh())

		return G


	def train(self,config):

		if config.dataset == 'mnist':
			(X_train, y_train), (X_test, y_test) = load_mnist()
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



		batches = int(len(X_train)/config.batch_size)		#int always rounds down --> no problem with running out of data

		counter = 1

		for epoch in range(config.epochs):
			for batch in range(config.batches):

				batch_X_real = X_train[batch*config.batch_size:(batch+1)*config.batch_size]
				batch_z = np.random.multivariate_normal(0,1,size=(config.batch_size,config.z_dim))
				batch_X_fake = self.G.predict(z)
				batch_X = np.concatenate((batch_X_real,batch_X_fake))

				batch_yd = np.concatenate((np.ones((config.batch_size)),np.zeros((config.batch_size))))
				batch_yg = np.ones((config.batch_size))

				#maybe normalize values in X?


				#Update D network
				D_loss = self.D.train_on_batch(batch_X, batch_yd)

				#Update G network
				G_loss = self.GAN.train_on_batch(batch_z, batch_yg)

				#Update G network again according to https://github.com/carpedm20/DCGAN-tensorflow.git
				G_loss = self.GAN.train_on_batch(batch_z, batch_yg)


				#Save losses to vectors in order to plot


				#Print status and save images for each config.sample_freq iterations
				if np.mod(counter,config.sample_freq) == 0:

					
					print('Epoch: {}/{} | Batch: {}/{} | D-loss {} | G-loss {}'.format(epoch,config.epochs,batch,batches,D_loss,G_loss))


				counter += 1

		print('\n' * 2)
		print('Training finished!')








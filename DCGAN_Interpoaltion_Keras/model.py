from keras.models import Sequential
from keras.layers import Dense, Reshape, ReLU, LeakyReLU, BatchNormalization as BN, Dropout
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import SGD, Adam, RMSprop
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
	
		self.D = self.discriminator(config)
		self.G = self.generator(config)
		self.D.trainable = False

		self.GAN = Sequential()
		self.GAN.add(self.G)
		self.GAN.add(self.D)

		loss, opt = get_loss_opt(config,'GAN')
		
		self.GAN.compile(loss=loss, optimizer=opt)
		
		#print('GAN:')
		#self.GAN.summary()

	def discriminator(self,config):

		init = get_init(config)
		input_shape = (config.x_h,config.x_w,config.x_d)

		if config.dataset == 'mnist':
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
			
			loss, opt = get_loss_opt(config,'D')
			D.compile(loss=loss, optimizer=opt)

		elif config.dataset == 'lines':

			D = Sequential() 
			input_shape = (config.x_h,config.x_w,config.x_d)

			D.add(Conv2D(filters=config.df_dim*1, kernel_size=3, strides=2, input_shape=input_shape, padding='same', kernel_initializer='random_uniform'))
			D.add(BatchNormalization(momentum=0.9))
			D.add(LeakyReLU(alpha=0.2))
			D.add(Dropout(config.dropout))

			D.add(Conv2D(filters=config.df_dim*2, kernel_size=3, strides=2, padding='same',kernel_initializer='random_uniform'))
			D.add(BatchNormalization(momentum=0.9))
			D.add(LeakyReLU(alpha=0.2))
			D.add(Dropout(config.dropout))

			D.add(Conv2D(filters=config.df_dim*4, kernel_size=3, strides=2, padding='same',kernel_initializer='random_uniform'))
			D.add(BatchNormalization(momentum=0.9))
			D.add(LeakyReLU(alpha=0.2))
			D.add(Dropout(config.dropout))

			D.add(Conv2D(filters=config.df_dim*8, kernel_size=3, strides=2, padding='same',kernel_initializer='random_uniform'))
			D.add(BatchNormalization(momentum=0.9))
			D.add(LeakyReLU(alpha=0.2))
			D.add(Dropout(config.dropout))

			D.add(Flatten())
			D.add(Dense(1))
			D.add(Activation('sigmoid'))

			loss, opt = get_loss_opt(config,'D')
			D.compile(loss=loss, optimizer=opt)

		#print('D:')
		#D.summary()

		return D

	def generator(self,config):

		init = RandomNormal(stddev=0.02)
		if config.dataset == 'mnist':
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
		
		elif config.dataset == 'lines':

			G = Sequential() 
			depth = 128
			dim = 8


			G.add(Dense(units=dim*dim*depth,input_dim=config.z_dim))
			G.add(Activation('relu'))
			G.add(Reshape((dim, dim, depth)))
			G.add(UpSampling2D())

			# In: dim X dim X depth
			# Out: 2*dim X 2*dim X depth/2 

			G.add(Conv2D(filters=depth, kernel_size=3, padding='same'))
			G.add(BatchNormalization(momentum=0.9))
			G.add(Activation('relu'))
			G.add(UpSampling2D())

			G.add(Conv2D(filters=int(depth/2), kernel_size=3, padding='same'))
			G.add(BatchNormalization(momentum=0.9))
			G.add(Activation('relu'))

			G.add(Conv2D(filters=1,kernel_size=3,padding='same'))
			G.add(Activation('tanh'))

		#print('G:')
		#G.summary()

		return G


	def train(self,config):

		print_training_setup(config)

		if config.dataset == 'mnist':
			X_train = load_mnist()
			batches = int(len(X_train)/config.batch_size)
		elif config.dataset == 'lines':
			batches = config.lines_batches
		elif config.dataset == 'celebA':
			(X_train, y_train), (X_test, y_test) = load_celebA()
			batches = int(len(X_train)/config.batch_size)

		
		D_loss_vec = []
		G_loss_vec = []
		batch_vec = []

		D_loss_vec.append(0)
		G_loss_vec.append(0)
		batch_vec.append(0)

		counter = 1

		print_training_initialized()

		start_time = time.time()
		t0 = start_time

		for epoch in range(config.epochs):
			for batch in range(batches):

				#Update D network
				if config.dataset == 'lines':
					batch_X_real, batch_yd_real= generate_real_samples_lines(config)
				elif config.dataset in ['mnist','celebA']:
					batch_X_real, batch_yd_real = generate_real_samples(config,X_train,random=config.random_sample,batch=batch)
				
				batch_X_fake, batch_yd_fake = generate_fake_samples(config,self.G)

				D_loss = train_D(config,self.D,batch_X_real,batch_yd_real,batch_X_fake,batch_yd_fake)

				#Update G network
				batch_z, batch_yg = generate_latent_codes(config)
				G_loss = train_G(config,self.GAN,batch_z,batch_yg)

				batch_z, batch_yg = generate_latent_codes(config)
				G_loss = train_G(config,self.GAN,batch_z,batch_yg)

				#Save losses to vectors in order to plot
				D_loss_vec.append(D_loss)
				G_loss_vec.append(G_loss)
				batch_vec.append(counter)

				#Save generated images
				if np.mod(counter,config.vis_freq) == 0 or (epoch==0 and batch==0):
					save_gen_imgs_batch(config,self.G,epoch,batch)

				#Plot training progress
				if np.mod(counter,config.plottrain_freq) == 0 and counter != 0:
					plot_save_train_prog(config,D_loss_vec,G_loss_vec,batch_vec,epoch,batch)

				#Print status and save images for each config.sample_freq iterations
				if np.mod(counter,config.progress_freq) == 0 and counter != 0:
					print_training_progress(config,epoch,batch,batches,D_loss,G_loss,start_time,t0)
					t0 = time.time()

				counter += 1

			#save model after each epoch
			save_model_checkpoint(config,epoch,batches,self.D,self.G,self.GAN)

		print_training_complete()
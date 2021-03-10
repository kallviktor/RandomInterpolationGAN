import os
import time
from pathlib import Path

class model_config(object):
	def __init__(self,
				 dataset='mnist',
				 loadmodel=False,
				 interpolation=False,
				 epochs=100,
				 batch_size=64,
				 lines_batches=1000,
				 z_dim=100,
				 z_start=0,
				 z_end=1,
				 int_time=1,
				 int_steps=10,
				 nmrParts=100,
				 thresh=0.95,
				 gf_dim=8,
				 gfc_dim=128,
				 dfc_dim=64,
				 c_dim=1,
				 optimizer=None,
				 loss_f=None,
				 learning_rate=0.0002,
				 beta_1=0.5,
				 init='RandomNormal',
				 init_stddev=0.02,
				 clip=0.01,
				 n_critic=5,
				 dropout=0.25,
				 progress_freq=200,
				 vis_freq=500,
				 plottrain_freq=500,
				 random_sample=False,
				 concatenate=False,
				 hm_xmin=0,
				 hm_xmax=2,
				 hm_ymin=0,
				 hm_ymax=2,
				 hm_steps=51,
				 out_dir='/out',
				 load_dir='/nodir'):

		"""
			Args:
				dataset:		[str]		Which dataset to use, ['mnist', 'lines', 'celebA'].
				loadmodel:		[bool]		loadmodel=True --> load pre trained model.
												loadmodel=False --> train a new model.
				interpolation:	[bool]		interpolation=True --> run the interpolation code.
												interpolation=False --> no interpolation.
				epochs:			[int]		Number of times the model is trained on the whole dataset.
				batch_size:		[int]		Number of datapoint in each batch of the training. Determines
											number of batches in one epoch.
				lines_batches:	[int]		Since the 'lines' dataset is generated on the fly, the notion of epoch
											and number of batches in one epoch becomes nonsensical. Instead,
											lines_batche sets the number of batches in one epoch.
				z_dim:			[int]		Dimension of the latent space.
				z_start:		[int]		Start position of interpolation. Same dimension as z_dim.
				z_end:			[int]		End position of interpolation. Same dimension as z_dim.
				time:			[int]		Total elapsed time in interpolation, a tuning paramter essentially.
				steps:			[int]		Number of steops in interpolation, including start and end point.
				nmrParts:		[int]		Number of particles in the SMC step of interpolation.
				gf_dim:			[int]		Dimension of the first convolutional layer in G.
				df_dim:			[int]		Dimension of the first convolutional layer in D.
				gfc_dim:		[int]		Chanels of the first convolutional layer in G.
				dfc_dim:		[int]		Chanels of the first convolutional layer in D.
				c_dim:			[int]		Number of channels in the output image. Set 1 for grayscale, 3 for color.
				optimizer:		[str]		Optimizer used in training, ['Adam', 'RMSprop'].
				loss_f:			[str]		Loss function used in training, ['Goodfellow','Wasserstein','binary_crossentropy']
												Goodfellow: From original GAN paper (slightly modified)
												Wasserstein: From WGAN paper.
				learning_rate:	[float]		Learning rate parameter in the optimizer used in training.
				beta_1:			[float]		The exponential decay rate for the 1st moment estimates in the Adam optimizer.
				init:			[str]		Initializer for the kernel in networks.
				init_stddev:	[float]		Standard deviation for initializer. Used in RandomNormal initializer.
				clip:			[float]		Clip rate used in WGAN.
				n_critic:		[int]		Number of times the critic (D) is trained for each training of the generatior (G), WGAN.
				dropout:		[float]		Parameter used in dropout layer.
				progress_freq:	[int]		Frequency of console update of the training progress.
				vis_freq:		[int]		Frequency that G generates images that are saved during training.
				plottrain_freq:	[int]		Frequency that the D & G loss is plotted and saved during training.
				random_sample:	[bool]		Determines whether samples in batches are drawn deterministically or stochastically.
												random_sample=True --> stochastic.
												random_sample=False --> deterministic.
				concatenate:	[bool]		Determines whether D is trained once on a batch of both real and fake samples
											or twice on two half-batches, one real and one fake.
												concatenate=True --> one batch of both real and fake.
												concatenate=False --> two half-batches.
				hm_xmin:		[int]		Minimum x value in heatmap.
				hm_xmax:		[int]		Maximum x value in heatmap.
				hm_ymin:		[int]		Minimum y value in heatmap.
				hm_ymax:		[int]		Maximum y value in heatmap.
				hm_steps:		[int]		Resolution of heatmap. Heatmap will be of resolution hm_steps x hm_steps.
				out_dir:		[str]		The name of the folder in which images and models will end up during and after training.
				load_dir:		[str]		Directory from which to load pre-trained models.

			"""


		# dataset specific setup
		if dataset == 'mnist':

			self.dataset 	= 'mnist'
			self.x_w 		= 32		#upsampled from 28 in utils.py
			self.x_h 		= 32		#upsampled from 28 in utils.py
			self.x_d 		= 1

		elif dataset == 'lines':

			self.dataset 	= 'lines'
			self.x_w		= 32
			self.x_h 		= 32
			self.x_d 		= 1
		
		elif dataset == 'celebA':

			self.dataset 	= 'celebA'
			self.x_w 		= 32
			self.x_h 		= 32
			self.x_d 		= 3

		# general setup
		self.learning_rate 	= learning_rate
		self.beta_1 		= beta_1
		self.init 			= init
		self.init_stddev 	= init_stddev
		self.clip 			= clip
		self.n_critic 		= n_critic
		self.dropout 		= dropout
		self.optimizer 		= optimizer
		self.loss_f 		= loss_f
		self.lines_batches 	= lines_batches
		self.random_sample 	= random_sample
		self.concatenate 	= concatenate
		self.progress_freq 	= progress_freq
		self.vis_freq 		= vis_freq
		self.plottrain_freq = plottrain_freq
		self.interpolation 	= interpolation
		self.batch_size 	= batch_size
		self.epochs 		= epochs
		self.z_dim 			= z_dim
		self.z_start 		= z_start
		self.z_end 			= z_end
		self.int_time 		= int_time
		self.int_steps 		= int_steps
		self.nmrParts 		= nmrParts
		self.thresh			= thresh
		self.gf_dim 		= gf_dim
		self.gfc_dim 		= gfc_dim
		self.dfc_dim 		= dfc_dim
		self.c_dim 			= c_dim

		#heatmap setup
		self.hm_xmin 		= hm_xmin
		self.hm_xmax		= hm_xmax
		self.hm_ymin		= hm_ymin
		self.hm_ymax		= hm_ymax
		self.hm_steps		= hm_steps

		# create/load model specific setup
		self.curr_dir 		= os.getcwd()
		self.out_dir 		= self.curr_dir+out_dir 

		if loadmodel:

			self.loadmodel 	= True
			self.load_dir 	= load_dir
			self.save_dir 	= str(Path(load_dir).parent.parent)
			self.inter_dir 	= self.save_dir+'/interpolation'
			self.hm_dir			= self.save_dir+'/heatmap'

		else:

			self.loadmodel 	= False
			self.save_dir 	= self.out_dir+'/'+time.strftime('%Y%m%d-%H%M')+'_'+self.dataset
			self.models_dir = self.save_dir+'/models'
			self.images_dir = self.save_dir+'/imgs'
			self.inter_dir 	= self.save_dir+'/interpolation'
			self.hm_dir			= self.save_dir+'/heatmap'
		
import os
import time

class model_config(object):
	def __init__(self, dataset='mnist',loadmodel=False,interpolation=False,epochs=100,batch_size=64,
						z_dim=100,gf_dim=64,df_dim=64,gfc_dim=1024,dfc_dim=1024,c_dim=1,learning_rate=0.0002,
						beta_1 = 0.5,progress_freq=200,vis_freq=500,plottrain_freq=500,
						out_dir='/out',load_dir='/nodir'):

		"""
			Args:
			sess: TensorFlow session
			batch_size: The size of batch. Should be specified before training.
			y_dim: (optional) Dimension of dim for y. [None]
			z_dim: (optional) Dimension of dim for Z. [100]
			gf_dim: (optional) Dimension of   [64]
			df_dim: (optional) Dimension of D filters in first conv layer. [64]
			gfc_dim: (optional) Dimension of G units for for fully connected layer. [1024]
			dfc_dim: (optional) Dimension of D units for fully connected layer. [1024]
			c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
		"""


		# dataset specific setup
		if dataset == 'mnist':

			self.dataset = 		'mnist'
			self.x_w = 			32		#upsampled from 28 in utils.py
			self.x_h = 			32		#upsampled from 28 in utils.py
			self.x_d = 			1

		elif dataset == 'lines':

			self.dataset = 		'lines'
			self.x_w = 			32
			self.x_h = 			32
			self.x_d = 			1
		
		elif dataset == 'celebA':

			self.dataset = 		'celebA'
			self.x_w = 			32
			self.x_h = 			32
			self.x_d = 			3



		


		# general setup
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.progress_freq = progress_freq
		self.vis_freq = vis_freq
		self.plottrain_freq = plottrain_freq
		self.interpolation = interpolation
		self.batch_size = 	batch_size
		self.epochs = 		epochs
		self.z_dim = 		z_dim
		self.gf_dim = 		gf_dim
		self.df_dim = 		df_dim
		self.gfc_dim = 		gfc_dim
		self.dfc_dim = 		dfc_dim
		self.c_dim = 		c_dim



		# create/load model specific setup
		self.curr_dir = os.getcwd()
		self.out_dir = 	self.curr_dir+out_dir 

		if loadmodel:

			self.loadmodel = True
			self.load_dir = load_dir

		else:

			self.loadmodel = False
			self.save_dir = self.out_dir+'/'+time.strftime('%Y%m%d-%H%M')+'_'+self.dataset
			self.models_dir = self.save_dir+'/models'
			self.images_dir = self.save_dir+'/imgs'
			self.interpolation_dir = self.save_dir+'/interpolation'
		
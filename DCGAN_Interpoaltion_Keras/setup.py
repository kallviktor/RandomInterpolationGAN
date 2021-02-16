class model_config(object):
	def __init__(self, dataset='mnist',loadmodel=False,,interpolation=False,epochs=100,batch_size=64,
						z_dim=100,gf_dim=64,df_dim=64,gfc_dim=1024,dfc_dim=1024):

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
			self.x_w = 			28
			self.x_h = 			28
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



		# create/load model specific setup
		if loadmodel:

			self.loadmodel = True
			self.load_dir = 'put dir here'

		else:

			self.loadmodel = False
			self.save_dir = 'put dir here'


		# general setup

		self.batch_size = 	batch_size
		self.epochs = 		epochs
		self.z_dim = 		z_dim
		self.gf_dim = 		gf_dim
		self.df_dim = 		df_dim
		self.gfc_dim = 		gfc_dim
		self.dfc_dim = 		dfc_dim
		self.c_dim = 		c_dim

		self.out_dir = 		'put dir here' 
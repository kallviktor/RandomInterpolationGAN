"""Storage file for networks that for varying reasons did not perform as expected."""

#====================================================

#Inspiration drawn from https://github.com/carpedm20/DCGAN-tensorflow.
#This repo was intended to further improve on the theory from the original DCGAN paper (arXiv:1511.06434) by
#training G twice for each D training batch.
#Did not converge. No further research to why this happened has been done.

#-------D-------
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

#-------G-------

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

#====================================================


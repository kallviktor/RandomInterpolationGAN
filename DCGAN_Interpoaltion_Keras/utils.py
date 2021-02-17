from keras.datasets import mnist
import math
import numpy as np

def load_mnist():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	X_train = np.pad(X_train, ((0,0),(2,2),(2,2)), 'constant')
	X_test = np.pad(X_test, ((0,0),(2,2),(2,2)), 'constant')

	return (X_train, y_train), (X_test, y_test)


def load_lines():
	pass


def load_celebA():
	pass


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

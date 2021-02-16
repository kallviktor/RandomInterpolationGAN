from keras.datasets import mnist

def load_mnist():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	return (X_train, y_train), (X_test, y_test)


def load_lines():
	pass


def load_celebA():
	pass

from keras import backend as K
from keras.engine.topology import layer
import numpy as np

class UpConvolution2D(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(UpConvolution2D, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[1]
		initial_weight_value = np.random.random((input_dim, output_dim))
		self.W = K.variable(initial_weight_value)
		self.trainable_weights = [self.W]

	def call(self, x, mask=None):
		return K.dot(x, self.W)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.output_dim)

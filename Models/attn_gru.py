import tensorflow as tf
from tensorflow.python.ops.nn import tanh
import math


def _get_dims(shape):
	fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
	fan_out = shape[1] if len(shape) == 2 else shape[-1]
	return fan_in, fan_out

def weight(name, shape, init='he', range=None):
	""" Initializes weight.
	:param name: Variable name
	:param shape: Tensor shape
	:param init: Init mode. xavier / normal / uniform / he (default is 'he')
	:param range:
	:return: Variable
	"""
	initializer = tf.constant_initializer()
	if init == 'xavier':
		fan_in, fan_out = _get_dims(shape)
		range = math.sqrt(6.0 / (fan_in + fan_out))
		initializer = tf.random_uniform_initializer(-range, range)

	elif init == 'he':
		fan_in, _ = _get_dims(shape)
		std = math.sqrt(2.0 / fan_in)
		initializer = tf.random_normal_initializer(stddev=std)

	elif init == 'normal':
		initializer = tf.random_normal_initializer(stddev=0.1)

	elif init == 'uniform':
		if range is None:
			raise ValueError("range must not be None if uniform init is used.")
		initializer = tf.random_uniform_initializer(-range, range)

	var = tf.get_variable(name, shape, initializer=initializer)
	return var

def bias(name, dim, initial_value=0.0):
	""" Initializes bias parameter.
	:param name: Variable name
	:param dim: Tensor size (list or int)
	:param initial_value: Initial bias term
	:return: Variable
	"""
	dims = dim if isinstance(dim, list) else [dim]
	return tf.get_variable(name, dims, initializer=tf.constant_initializer(initial_value))

class AttnGRU:
    # Attention-based Gated Recurrent Unit cell (cf. https://arxiv.org/abs/1603.01417).
	
	def __init__(self, num_units):
		self._num_units = num_units
	
	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units
	
	def __call__(self, inputs, state, attention, scope=None):
		# Gated recurrent unit (GRU) with nunits cells.
		with tf.variable_scope(scope or 'AttrGRU'):
			with tf.variable_scope("Gates"):  # Reset gate and update gate.
				# We start with bias of 1.0 to not reset.
				r = tf.nn.sigmoid(self._linear(inputs, state, bias_default=1.0))
			with tf.variable_scope("Candidate"):
				c = tanh(self._linear(inputs, r * state))
	
			new_h = tf.expand_dims(attention, -1) * c + (1 - tf.expand_dims(attention, -1)) * state
		return new_h
						
	def _linear(self, x, h, bias_default=0.0):
		I, D = x.get_shape().as_list()[1], self._num_units
		w = weight('W', [I, D])
		u = weight('U', [D, D])
		b = bias('b', D, bias_default)
															
		return tf.matmul(x, w) + tf.matmul(h, u) + b

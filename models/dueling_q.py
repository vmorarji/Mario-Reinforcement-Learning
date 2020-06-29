"""Dueling-Q Network"""
import tensorflow as tf
from tensorflow import keras

class DQModel(keras.Model):
	## Create a Dueling-Q learning nn
	def __init__(self, hidden_size: int, num_actions: int):
		"""
		Args:
			hidden_state: The size of the hidden layer
			num_actions: The size of the output layer. Normallly the same
				as the environments number of actions.
		"""
		super(DQModel, self).__init__()
		self.conv1 = keras.layers.Conv2D(32, (8, 8), (4, 4), activation='relu')
		self.conv2 = keras.layers.Conv2D(64, (4, 4), (2, 2), activation='relu')
		self.conv3 = keras.layers.Conv2D(64, (4, 4), (2, 2), activation='relu')
		self.flatten = keras.layers.Flatten()
		self.adv_dense = keras.layers.Dense(hidden_size, activation='relu',
			kernel_initializer=keras.initializers.he_normal())
		self.adv_out = keras.layers.Dense(num_actions,
			kernel_initializer=keras.initializers.he_normal())
		self.v_dense = keras.layers.Dense(hidden_size, activation='relu',
			kernel_initializer=keras.initializers.he_normal())
		self.v_out = keras.layers.Dense(1, 
			kernel_initializer=keras.initializers.he_normal())
		self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
		self.combine = keras.layers.Add()

	def call(self, input):
		x = self.conv1(input)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		adv = self.adv_dense(x)
		adv = self.adv_out(adv)
		v = self.v_dense(x)
		v = self.v_out(v)
		norm_adv = self.lambda_layer(adv)
		combined = self.combine([v, norm_adv])
		return combined
		return adv
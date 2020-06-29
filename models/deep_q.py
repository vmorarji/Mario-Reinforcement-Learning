"""Deep-Q Network"""
import tensorflow as tf
from tensorflow import keras

class DQModel(keras.Model):
	## Create a Q Learning nn
	def __init__(self, hidden_size: int, num_actions: int):
		"""
		Args:
			hidden_state: The size of the hidden layer
			num_actions: The size of the output layer. Normally be the same
				as the environments number of actions.
		"""
		super(DQModel, self).__init__()
		self.conv1 = keras.layers.Conv2D(32, (8, 8), (4, 4), activation='relu')
		self.conv2 = keras.layers.Conv2D(64, (4, 4), (2, 2), activation='relu')
		self.conv3 = keras.layers.Conv2D(64, (4, 4), (2, 2), activation='relu')
		self.flatten = keras.layers.Flatten()
		self.dense1 = keras.layers.Dense(hidden_size, activation='relu',
			kernel_initializer=keras.initializers.he_normal())
		self.out = keras.layers.Dense(num_actions,
			kernel_initializer=keras.initializers.he_normal())

	def call(self, input):
		x = self.conv1(input)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.out(x)

		return x
"""Methods for training an Agent to play Mario."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime as dt
import random
import os
from models import deep_q, dueling_q
from replaybuffer import replay
from replaybuffer import per
from preprocess.preprocess_image import image_reshape

class Evaluate:

	"""Initialise the networks and memory buffer to train an agent to play Mario."""
	def __init__(
		self, 
		observation_space,
		num_actions: int,
		num_frames: int,
		delay_timesteps: int = 1e4,
		max_policy_error: float = 100.0,
		max_beta: float = 1.0,
		min_beta: float = 0.4,
		beta_decay_iter: int = 45e4,
		max_epsilon: float = 1.0,
		min_epsilon: float = 0.01,
		epsilon_decay_iter: int = 45e4,
		hidden_size: int = 256,
		learning_rate: float = 1e-3,
		gamma: int = 0.99,
		tau: int = 0.05,
		loss_fn = None,
		memory_size: int = 131072,
		batch_size: int = 64,
		summaries: bool = False,
		summary_iter: int = 100,
		current_time = None,
		dueling: bool = False, 
		per_memory: bool = True,
		save_model: bool = False,
		save_model_freq: int = 5e4
		):
		"""
		Args:
			observation_space: The post process resolution of one state/frame of
				 the environment as a tuple.
			num_actions: The total number of actions the agent can use to play 
				the environment.
			num_frames: The number of sequential states that will be stacked.
				The networks will use stacked states for training and action 
				selection.
			delay_timesteps: The number of timesteps that will be played 
				entirely at random by the agent. A delay is to ensure that the
				replay buffer is sufficiently filled before training starts.
			max_policy_error: The maximum error that an experience tuple can 
				have in the PER memory buffer.
			max_beta: The maximum value that beta will be gradually adjusted 
				towards. 
			min_beta: The initial value of beta.
			beta_decay_iter. The number of timesteps for beta to linearly
				increase from min_beta to max_beta. Beta will be used to create
				the weights for training the model.
			max_epsilon: The initial value of epsilon.
			min_epsilon: The minimum value that epsilon will be decayed towards
				during training.
			epsilon_decay_iter: The number of timesteps that epsilon will
				linearly decay from max_epsilon to min_epsilon.
			hidden_size: The size of the hidden layer(s) in the neural networks.
			learning_rate: The learning rate of the Adam optimizer. For PER 0.0001
				is recommended.
			gamma: The discount factor for Q(s’,a’) in Q-learning.
			tau: The rate that the target network will update from the primary 
				network.
			loss: A keras or tensorflow loss object. Huber is the default loss.
			memory_size: The total size of the replay buffer. For PER the memory
				size must be in the geometric series 2^n for the SumTree to
				initialise with all the leaf nodes.
			batch_size: The size of the batch used for training the neural 
				networks.
			summaries: True to store tensorboard summaries of training.
			summary_iter: The timestep interval that logs will be made of the 
				tensorboard summaries.
			current_time: The current time used for creating folders for both 
				the model variables and the summary logs. If none the current
				time will be used
			dueling: If True a dueling DQN will be initialised. If False
				DDQN will be used.
			per_memory: If True performance experience replay will be used as
				the memory buffer. If False a standard replay buffer will be
				initialised.
			save_model: True will save the weights of the models as checkpoints.
			save_model_freq: How often the models’ weights will be saved.
		"""

		self.observation_space = observation_space
		self.num_actions = num_actions
		self.num_frames = num_frames
		self.delay_timesteps = delay_timesteps
		self.max_policy_error = max_policy_error
		self.beta = min_beta
		self.max_beta = max_beta
		self.min_beta = min_beta
		self.beta_decay_iter = beta_decay_iter
		self.epsilon = max_epsilon
		self.max_epsilon = max_epsilon
		self.min_epsilon = min_epsilon
		self.epsilon_decay_iter = epsilon_decay_iter
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.batch_size = batch_size
		self.tau = tau
		self.loss_fn = (loss_fn or tf.keras.losses.Huber())
		self.num_actions = num_actions
		self.summaries = summaries
		self.summary_iter = summary_iter
		self.dueling = dueling
		self.per_memory = per_memory
		self.save_model = save_model
		self.save_model_freq = save_model_freq

		# initialise the network
		if self.dueling:
			self.primary_network = dueling_q.DQModel(self.hidden_size, self.num_actions)
			self.target_network = dueling_q.DQModel(self.hidden_size, self.num_actions)
		else:
			self.primary_network = deep_q.DQModel(self.hidden_size, self.num_actions)
			self.target_network = deep_q.DQModel(self.hidden_size, self.num_actions)

		for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables):
			t.assign(e)

		self.primary_network.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss=self.loss_fn)
	
		# initialise the memory
		if self.per_memory:
			self.memory = per.Memory(self.observation_space, self.num_frames, memory_size, self.max_policy_error)
		else:
			self.memory = replay.Memory(self.observation_space, self.num_frames, memory_size)

		# create a summary writer
		if current_time is None:
			self.current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
		else:
			self.current_time = current_time

		if self.summaries:
			self.train_writer = tf.summary.create_file_writer('./logs/' + self.current_time)

		if self.save_model:
			if not os.path.exists("./saved_models/{}".format(self.current_time)):
				os.makedirs("./saved_models/{}".format(self.current_time))


	def train(self, steps):
		""" Train the neural networks and step the epsilon and beta values
		Args:
			steps: The total timesteps taken by the agent up to the point 
				of the current training step.
		"""

		if steps < self.delay_timesteps:
			return -1

		if self.per_memory:
			states, actions, rewards, next_states, terminal, idxs, is_weights = self.memory.sample(self.batch_size)
		else:
			states, actions, rewards, next_states, terminal = self.memory.sample(self.batch_size)
			is_weights = np.ones(self.batch_size)
		target_q, error = self.evaluate_network(states, actions, rewards, next_states, terminal)

		if self.per_memory:
			for i in range(self.batch_size):
				self.memory.update(idxs[i], error[i])
			if (steps - self.delay_timesteps) < self.beta_decay_iter:
				self.beta = self.min_beta + ((steps - self.delay_timesteps) / self.beta_decay_iter) \
				* (self.max_beta - self.min_beta)
			else:
				self.beta = self.max_beta
			self.memory.beta = self.beta

		loss = self.primary_network.train_on_batch(states, target_q, is_weights)

		# update target network parameters slowly from primary network
		for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables):
			t.assign(t * (1 - self.tau) + e * self.tau)

		if self.summaries:
			if steps % self.summary_iter == 0:
				with self.train_writer.as_default():
					tf.summary.scalar('target_q_mean', tf.reduce_mean(target_q), step=steps)
					tf.summary.scalar('target_q_max', tf.reduce_max(target_q), step=steps)
					tf.summary.scalar('target_q_min', tf.reduce_min(target_q), step=steps)

					tf.summary.scalar('td_error_mean', tf.reduce_mean(error), step=steps)
					tf.summary.scalar('td_error_max', tf.reduce_max(error), step=steps)
					tf.summary.scalar('td_error_min', tf.reduce_min(error), step=steps)

					tf.summary.histogram('target_q_hist', target_q, step=steps)
					tf.summary.histogram('td_error_hist', error, step=steps)
					tf.summary.histogram('target_q_argmax', np.argmax(target_q, axis=1), step=steps)

		if (steps - self.delay_timesteps) < self.beta_decay_iter:
			self.epsilon = self.max_epsilon - ((steps - self.delay_timesteps) / self.epsilon_decay_iter) \
			* (self.max_epsilon - self.min_epsilon)
		else:
			self.epsilon = self.min_epsilon

		if steps % self.save_model_freq == 0:
			self.save_networks(steps)

		return loss

	def evaluate_network(self, states, actions, rewards, next_states, terminal):
		""" Calculate the Q-value and the error for a state or a set of states
		Args:
			states: A state directly from the environment or set of states from 
				the replay buffer.
			actions: An action directly from the environment or set of actions 
				from the replay buffer.
			rewards: A reward directly from the environment or set of rewards 
				from the replay buffer.
			next_states: The next state directly from the environment or set of
				 next states from the replay buffer.
			terminal: A Boolean value if this was a terminal transition or a 
				set of Boolean values from the replay buffer.
		"""

		# preditct Q(s,a) given the batch states
		prim_qt = self.primary_network(states)
		# predict Q(s', a') from the evaluation network
		prim_qtp1 = self.primary_network(next_states)
		# copy the prim_qt tensor into the target_q tensor
		target_q = prim_qt.numpy()
		# a' selection
		prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
		# the q value for prim_action_tp1 comes from the target network
		q_from_target = self.target_network(next_states)
		updates = rewards + (1 - terminal) * self.gamma * \
		q_from_target.numpy()[np.arange(0, states.shape[0]), prim_action_tp1]
		# update the action index of target_q with the update
		target_q[np.arange(0, states.shape[0]), actions] = updates
		# calculate the loss/error to update priorities
		error = []
		for i in range(states.shape[0]):
			error.append(self.loss_fn(target_q[i, actions[i]], prim_qt.numpy()[i, actions[i]]))

		return target_q, error



	def choose_action(self, state, steps):
		""" Choose an action based on the Epsilon Greedy policy.
		Args:
			state: The current state of the environment after any processing.
			steps: The current total timesteps that the agent has played.
		"""

		if random.random() < self.epsilon:
			return np.random.randint(0, self.num_actions)
		else:
			action_state = image_reshape(state, self.observation_space, self.num_frames)
			return self.primary_network.predict(action_state).argmax()

	def save_networks(self, steps):
		""" Save the networks.
		Args:
			Steps: The current total timesteps that the agent has played. 
		"""

		self.primary_network.save_weights("./saved_models/{}/primary_network_{}".format(self.current_time, steps))
		self.target_network.save_weights("./saved_models/{}/target_network_{}".format(self.current_time, steps))
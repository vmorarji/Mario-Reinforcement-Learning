""" A replay buffer to store and retrieve experience tuples. """

import numpy as np

class Memory(object):
	""" A replay buffer for storing experience tuples."""
	def __init__(self, image_size, num_frames: int, size: int):
		"""
		Args:
			image_size: The size of the frame post processing. The
				default size for super_mario_bros_py is 256x240. 
			num_frames: The number of sequential states that will
				be stacked together when sampled.
			size: The total size of the memory buffer. Once full
				the write index will reset and the oldest experience
				tuple will be replaced.
		"""

		self.size = size
		self.curr_write_idx = 0
		self.available_samples = 0
		self.image_size = image_size
		self.num_frames = num_frames
		self.buffer = [(np.zeros((self.image_size[0], self.image_size[1]), 
			dtype=np.float32), 0.0, 0.0, 0.0) for i in range(self.size)]
		self.frame_idx = 0
		self.action_idx = 1
		self.reward_idx = 2
		self.terminal_idx = 3

	def append(self, experience: tuple):
		""" Save a new experience tuple into the memory buffer.
		Args:
			experience: A tuple that is in the order –
			next_state, action, reward, terminal
		"""

		self.buffer[self.curr_write_idx] = experience
		self.curr_write_idx += 1
		# reset the current writer position if greater than the allowed size
		if self.curr_write_idx >= self.size:
			self.curr_write_idx = 0
		# max out available samples at the memory buffer size
		if self.available_samples + 1 < self.size:
			self.available_samples += 1
		else:
			self.available_samples = self.size - 1

	def sample(self, num_samples: int):
		""" Return a sample of random experience tuples from the replay buffer.
		Args:
			num_samples: The number of experience tuples that will
				be drawn from the replay buffer. 
		"""

		sampled_idxs = np.random.randint(self.num_frames, self.available_samples, num_samples)
		# now load up the state and next state variables according to sampled idxs
		states = np.zeros((num_samples, 
							self.image_size[0],
							self.image_size[1],
							self.num_frames),
							dtype=np.float32)
		next_states = np.zeros((num_samples,
								self.image_size[0],
								self.image_size[1],
								self.num_frames),
								dtype=np.float32)
		actions, rewards, terminal = [], [], []
		for i, idx in enumerate(sampled_idxs):
			for j in range(self.num_frames):
				states[i, :, :, j] = self.buffer[idx + j - self.num_frames][self.frame_idx][:, :, 0]
				next_states[i, :, :, j] = self.buffer[idx + j - self.num_frames + 1][self.frame_idx][:, :, 0]
			actions.append(self.buffer[idx][self.action_idx])
			rewards.append(self.buffer[idx][self.reward_idx])
			terminal.append(self.buffer[idx][self.terminal_idx])
		return states, np.array(actions), np.array(rewards), next_states, np.array(terminal)
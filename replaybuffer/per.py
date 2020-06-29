""" A prioritised experience replay buffer to store and retrieve
	experience tuples from and a SumTree to draw values based on
	a specified priority
"""

import numpy as np

class Node:
	""" The SumTree that all priorities will be stored and used to draw samples """
	def __init__(self, left, right, is_leaf: bool = False, idx = None):
		"""
		Args:
			left: The left node for the SumTree. The value is added
				to the right node for the value of their parent node.
			right: The right node for the SumTree. The value is added
				to the left ndoe for the value of their parent node.
			is_leaf: A boolean value to state whether the current
				nodes are leaf nodes.
			idx: The index of the nodes within the SumTree
		"""

		self.left = left
		self.right = right
		self.is_leaf = is_leaf
		self.value = sum(n.value for n in (left, right) if n is not None)
		self.parent = None
		self.idx = idx  # this value is only set for leaf nodes
		if left is not None:
			left.parent = self
		if right is not None:
			right.parent = self

	@classmethod
	def create_leaf(cls, value, idx):
		leaf = cls(None, None, is_leaf=True, idx=idx)
		leaf.value = value
		return leaf


	def retrieve(self, value: float):
		""" Retrieve a leaf node from the SumTree
		Args:
			value: A number that is less than the value of the base node.
		Returns:
			A leaf node. The sum of the values of this node and all the 
				nodes that preceded will be equivalent to the argument value.
		"""

		if self.is_leaf:
			return self

		if self.left.value >= value: 
			return self.left.retrieve(value)
		else:
			value -= self.left.value
			return self.right.retrieve(value)

	def update(self, new_value: float):
		""" Update one index with a new value and propagate the changes 
			throughout the SumTree.
		Args:
			value: The new value of the node.
		"""

		change = new_value - self.value

		self.value = new_value
		self.parent.propagate_changes(change)


	def propagate_changes(self, change: float):
		""" Apply the change in value of a node throughout the network.
		Args:
			change: The change in value that will be added or subtracted
				from the parent of the current node.
		"""

		self.value += change

		if self.parent is not None:
			self.parent.propagate_changes(change)


class Memory(object):
	"""An experience replay used to store experience tuples and draw samples based on a SumTree"""
	def __init__(self, image_size, num_frames: int, size: int, max_priority: float):
		"""
		Args:
			image_size: The size of the frame post processing. The default 
				size for super_mario_bros_py is 256x240. 
			num_frames: The number of sequential states that will be stacked
				together when sampled.
			size: The total size of the memory buffer. Once full the write
				index will reset and the oldest experience tuple will be 
				replaced. The total size needs to be in the geometric series 
				2^n for the SumTree to initialise.
			max_priority: The maximum value a leaf node can have. This 
				limits extraordinary large values being overly sampled.
		"""
		if np.log2(size) % 1 !=0:
			raise ValueError('memory size should be in the geometric series 2^n')
		self.curr_write_idx = 0
		self.available_samples = 0
		self.image_size = image_size
		self.num_frames = num_frames
		self.size = size
		self.buffer = [(np.zeros((self.image_size[0], self.image_size[1]),
			dtype=np.float32), 0.0, 0.0, 0.0) for i in range(self.size)]
		self.base_node, self.leaf_nodes = create_tree([0.0 for i in range(self.size)])
		self.frame_idx = 0
		self.action_idx = 1
		self.reward_idx = 2
		self.terminal_idx = 3
		self.beta = 0.4
		self.alpha = 0.6
		self.min_priority = 0.01
		self.max_priority = max_priority

	def append(self, experience: tuple, priority: float):
		""" Add an experience tuple to the replay buffer
		Args:
			experience: A tuple that is in the order – 
				next_state, action, reward, terminal
			priority: The priority value of the experience tuple. The 
				priority will be added to the leaf node of the SumTree.
		"""

		self.buffer[self.curr_write_idx] = experience
		self.update(self.curr_write_idx, priority)
		self.curr_write_idx += 1
		# reset the current writer position if greater than the allowed size
		if self.curr_write_idx >= self.size:
			self.curr_write_idx = 0
		# max out available sampoles at the memory buffer size
		if self.available_samples + 1 < self.size:
			self.available_samples += 1
		else:
			self.available_samples = self.size - 1

	def update(self, idx: int, priority: float):
		""" Update one index with a new value and propagate the 
			changes throughout the SumTree.
		Args:
			idx: The index of the leaf node to be updated.			
			priority: The new value of the node.
		"""
		self.leaf_nodes[idx].update(self.adjust_priority(priority))
        
	def adjust_priority(self, priority: float):
		""" Adjust the priority of one node by adding a minimum value to 
			ensure there is some probability of drawing every sample.
		Args:
			priority: The new value of the node.
		"""

		adj_priority = np.power(priority + self.min_priority, self.alpha)
		if adj_priority > self.max_priority:
			return self.max_priority
		else:
			return adj_priority
    
	def sample(self, num_samples: int):
		"""Use the SumTree to draw samples from the replay buffer
		Args:
			num_samples: The number of experience tuples to be returned
		Returns:
			An np.array of states, actions, rewards, next_states and 
				terminals. The experience tuples’ indexes are also returned 
				along with the weights for learning.
		"""

		sampled_idxs = []
		is_weights = []
		sample_no = 0
		while sample_no < num_samples:
			sample_val = np.random.uniform(0, self.base_node.value)
			samp_node = self.base_node.retrieve(sample_val)
			if self.num_frames - 1 < samp_node.idx < self.available_samples - 1:
				sampled_idxs.append(samp_node.idx)
				p = samp_node.value / self.base_node.value
				is_weights.append((self.available_samples + 1) * p)
				sample_no += 1
        # apply the beta factor and normalise so that the maximum is_weight < 1
		is_weights = np.array(is_weights)
		is_weights = np.power(is_weights, -self.beta)
		is_weights = is_weights / np.max(is_weights)
		# now load up the state and enxt state variables according to sampled idxs
		states = np.zeros((num_samples, self.image_size[0], self.image_size[1], self.num_frames),
							dtype=np.float32)
		next_states = np.zeros((num_samples, self.image_size[0], self.image_size[1], self.num_frames),
								dtype=np.float32)
		actions, rewards, terminal = [], [], []
		for i, idx in enumerate(sampled_idxs):
			for j in range(self.num_frames):
				states[i, :, :, j] = self.buffer[idx + j - self.num_frames + 1][self.frame_idx][:, :, 0]
				next_states[i, :, :, j] = self.buffer[idx + j - self.num_frames + 2][self.frame_idx][:, :, 0]
			actions.append(self.buffer[idx][self.action_idx])
			rewards.append(self.buffer[idx][self.reward_idx])
			terminal.append(self.buffer[idx][self.terminal_idx])
		return states, np.array(actions), np.array(rewards), next_states, np.array(terminal), sampled_idxs, is_weights


def create_tree(input: list):
	""" Create a SumTree by iterating through the class function create_leaf.
	Args:
		Input: A list of values for the leaf nodes.
	"""

	nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
	leaf_nodes = nodes
	while len(nodes) > 1:
		inodes = iter(nodes)
		nodes = [Node(*pair) for pair in zip(inodes, inodes)]

	return nodes[0], leaf_nodes
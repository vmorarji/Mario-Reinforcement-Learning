import tensorflow as tf


def image_preprocess(image, new_size=(75, 80)):
	# normalise the image
	image = tf.image.rgb_to_grayscale(image)
	image = tf.image.resize(image, new_size)
	image = tf.cast(image / 255, dtype=tf.float32)
	return image


def process_state_stack(state_stack, state):
	# update the state stack with a new state
	for i in range(1, state_stack.shape[-1]):
		state_stack[:, :, i - 1].assign(state_stack[:, :, i])
	state_stack[:, :, -1].assign(state[:, :, 0])
	return state_stack


def image_reshape(state_stack, image_size, num_frames):
	# resize the state_stack for the NN
	return tf.reshape(state_stack, (1, image_size[0], image_size[1], num_frames))
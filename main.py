import numpy as np
import tensorflow as tf
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import argparse
import os
import datetime as dt

from train import Evaluate
from preprocess.wrappers import wrapper
from preprocess.preprocess_image import image_preprocess
from preprocess.preprocess_image import image_reshape
from preprocess.preprocess_image import process_state_stack

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="SuperMarioBros-1-1-v0", help='Gym environment and level')
	parser.add_argument("--dueling", action='store_true', help='True for Dueling DQN, False for Double DQN')
	parser.add_argument("--per", action='store_true', help='True for Performance Experience Replay')
	parser.add_argument("--frame_size", default=(84, 84), type=tuple, help='Resize the frame to decrease computational load')
	parser.add_argument("--max_timesteps", default=5e5, type=int, help='Total time steps the environment will play')
	parser.add_argument("--delay_timesteps", default=1e4, type=int, help='Initial time steps that have random actions')
	parser.add_argument("--min_epsilon", default=0.01, type=float, help='The lowest value for epsilon')
	parser.add_argument("--epsilon_decay", default=45e4, type=int, help='Time steps required to reach the minimum epsilon')
	parser.add_argument("--beta_decay", default=45e4, type=int, help='Time steps required to reach the maximum beta')
	parser.add_argument("--learning_rate", default=1e-3, type=float, help='Learning rate for the optimizer')
	parser.add_argument("--gamma", default=0.99, type=float, help="Gamma discount for Q(s',a')")
	parser.add_argument("--tau", default=0.05, type=float, help='Tau rate for updating the target network')
	parser.add_argument("--memory_size", default=131072, type=int, help='Size of the memory buffer. PER requires 2^x')
	parser.add_argument("--batch_size", default=64, type=int, help='The size of the mini-batches for Q learning')
	parser.add_argument("--debug_summary", action='store_true', help='True for tensorboard summaries of training')
	parser.add_argument("--save_model", action='store_true', help='Store the model')
	parser.add_argument("--save_freq", default=5e4, type=int, help="How often the models' weights are saved")
	args = parser.parse_args()


	# Create a store path for results and debug_summaries
	save_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

	reward_writer = tf.summary.create_file_writer('./logs/' + save_time)

	# Initialise the environment
	env = gym_super_mario_bros.make(args.env)
	env = JoypadSpace(env, RIGHT_ONLY)
	env = wrapper(env)

	num_actions = env.action_space.n
	observation_space = args.frame_size
	num_frames = 4

	# Initialise the agent
	kwargs = {
		"observation_space": observation_space,
		"num_actions": num_actions,
		"num_frames": num_frames,
		"delay_timesteps": args.delay_timesteps,
		"beta_decay_iter": args.beta_decay,
		"min_epsilon": args.min_epsilon,
		"epsilon_decay_iter": args.epsilon_decay,
		"learning_rate": args.learning_rate,
		"gamma": args.gamma,
		"tau": args.tau,
		"memory_size": args.memory_size,
		"batch_size": args.batch_size,
		"summaries": args.debug_summary,
		"current_time": save_time,
		"dueling": args.dueling,
		"per_memory": args.per,
		"save_model": args.save_model,
		"save_model_freq": args.save_freq
	}


	agent = Evaluate(**kwargs)


	# Initialise the main run variables and start the training loop
	MAX_STEPS = args.max_timesteps
	done = True
	episode_num = 0
	episode_reward = 0
	total_timesteps = 0
	avg_loss = 0

	while total_timesteps < MAX_STEPS:
	
		# If an episode is over print the episode summary and reset the environment
		if done:
			if total_timesteps != 0:
				print('Episode: {}, Completion: {:.2f}%, Episode Reward: {:.2f}, loss: {:.3f}, eps: {:.3f}'.format(
					episode_num,
					total_timesteps / MAX_STEPS * 100,
					episode_reward,
					avg_loss / episode_timesteps,
					agent.epsilon,
				))

				with reward_writer.as_default():
					tf.summary.scalar('episode_reward', episode_reward, step=episode_num)

			state = env.reset()
			state = image_preprocess(state, observation_space)
			state_stack = tf.Variable(np.repeat(state, num_frames).reshape((observation_space[0],
																			observation_space[1],
																			num_frames)))
			

			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			avg_loss = 0

		# Choose and play the action through the epsilon greedy policy.
		action = agent.choose_action(state_stack, total_timesteps)
		next_state, reward, done, info = env.step(action)

		episode_reward += reward

		# process the state to the right size and update the state stack
		next_state = image_preprocess(next_state, observation_space)
		old_state_stack = state_stack
		state_stack = process_state_stack(state_stack, next_state)
		
		# train the networks
		loss = agent.train(total_timesteps)

		if agent.per_memory:
			# Evaluate the current state and use the error as the priority.
			# Append these values to the memory
			if total_timesteps > agent.delay_timesteps:
				processed_state = image_reshape(old_state_stack, observation_space, num_frames)
				processed_next_state = image_reshape(state_stack, observation_space, num_frames)
				_, error = agent.evaluate_network(processed_state,
												np.array([action]),
												np.array([reward]),
												processed_next_state,
												np.array([done])
												)
				agent.memory.append((next_state, action, reward, done), error[0])
				
			else:
				# During the training delay use the reward as a heuristic for the priority
				agent.memory.append((next_state, action, reward, done), reward if reward > 0 else 0)
		else:
			agent.memory.append((next_state, action, reward, done))


		total_timesteps += 1
		episode_timesteps += 1
		avg_loss += loss

	# Save the networks at the last timestep
	agent.save_networks(int(MAX_STEPS))
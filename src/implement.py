import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
from datetime import datetime
from GenRep import GenerativeReplay


if __name__ == "__main__":
	print("Starting...")
	
	# Hyper parameters

	# General
	USE_GENERATIVE = True
	NO_REPLAY = False
	RECORD_TRAINING_TIMES = False
	ENV = "MountainCarContinuous-v0"
	START_TIMESTEPS = 15e3
	END = START_TIMESTEPS + 50e5
	EVAL_FREQ = 5e3
	MAX_TIMESTEPS = 2e7
	SEED = 10
	FILE_NAME = "a"

	# TD3 parameters
	EXPL_NOISE = 0.1
	BATCH_SIZE = 128
	DISCOUNT = 0.99
	TAU = 0.005
	POLICY_NOISE = 0.2
	NOISE_CLIP = 0.5
	POLICY_FREQ = 2

	print_interval = 1000

	print(f"Start new process with {ENV} and file name {FILE_NAME}")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(ENV)

	# Set seeds
	env.seed(SEED)
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	
	# Some env dimentions
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	print(state_dim, action_dim, max_action)
	print(env.observation_space, env.action_space)
	print("GenerativeReplay: ", USE_GENERATIVE)

	# Build TD3
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": DISCOUNT,
		"tau": TAU,
		"policy_noise": POLICY_NOISE,# * max_action,
		"noise_clip": NOISE_CLIP,# * max_action,
		"policy_freq": POLICY_FREQ
	}

	policy = TD3.TD3(**kwargs)

	# Use optim buffer or not. Type 0 uses episode reward and 1 uses state as comparison
	OPTIMAL_BUFFER = True
	OPTIMAL_BUFFER_TYPE = 0

	print("Use Optim Buffer:", OPTIMAL_BUFFER)
	if OPTIMAL_BUFFER:
		if OPTIMAL_BUFFER_TYPE == 0:
			print("Buffer Type: Episode Reward")
		else:		
			print("Buffer Type: State")

	# Make the replay component
	replay_component = None
	if USE_GENERATIVE:
		replay_component = GenerativeReplay()
		if OPTIMAL_BUFFER:
			replay_component.use_optim_buffer = True
	elif NO_REPLAY:
		replay_component = utils.ReplayBuffer(state_dim, action_dim, 256)
		OPTIMAL_BUFFER = False
	else:
		replay_component = utils.ReplayBuffer(state_dim, action_dim)
		OPTIMAL_BUFFER = False
	

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	max_state = None
	evaluations = []

	TD3_training = False


	# During the initial exploration of the environment, achieving positive reward is extremely rare
	# In order see the capabilities (without having to re-run) of ER in a positive reward scenario this is implemented
	guarantee_finish = False
	gf_count = 1
	action = None
	gf = 0
	if guarantee_finish:
		action = [1.0]


	
		
	for t in range(int(MAX_TIMESTEPS)):
		# env.render()

		episode_timesteps += 1

		if t >= END:
			raise ValueError

		# Select action randomly or according to policy based on the start timesteps
		if t < START_TIMESTEPS:
			
			if guarantee_finish:
				if state[1] < 0.0001 and action[0] == 1.0:
					action = np.array([-1.0])
				elif state[1] > -0.0001 and action[0] == -1.0:
					action = np.array([1.0])
			else:
				action = env.action_space.sample()
			episode_num = 0
		else:
			if OPTIMAL_BUFFER and replay_component.highest_reward_state == None:
				replay_component.edit_optim_buffer(-10000)
			replay_component.training = True
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)

		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay component
		GAN_training = replay_component.add(state, action, next_state, reward, done_bool)
		
		if OPTIMAL_BUFFER and OPTIMAL_BUFFER_TYPE == 1:
			if max_state == None:
				max_state = -100
			elif state[0] > max_state:
				max_state = state[0]

		state = next_state
		episode_reward += reward
		

		
		# Train agent after collecting sufficient data
		if t >= START_TIMESTEPS:
			# env.render()
			if TD3_training:
				print("TD3 Training...")
				TD3_training = True
			policy.train(replay_component, BATCH_SIZE)
			
		if done: 
			if guarantee_finish:
				if episode_reward > 0:
					if gf == gf_count-1:
						guarantee_finish = False
					else:
						gf += 1
			if OPTIMAL_BUFFER:
				if replay_component.training and t >= START_TIMESTEPS:
					if OPTIMAL_BUFFER_TYPE == 0:
						if episode_reward < -10 or episode_reward > 0:
							if replay_component.highest_reward_state < episode_reward:
								print(episode_reward)
								replay_component.edit_optim_buffer(episode_reward)
					else:
						if replay_component.highest_reward_state < max_state:
							print(max_state)
							replay_component.edit_optim_buffer(max_state)
			


			print(f"Episode {episode_num}, reward is {episode_reward}, episode_timesteps {episode_timesteps}")
			if t >= START_TIMESTEPS:
				evaluations.append(episode_reward)
				np.save(f"./results/{FILE_NAME}", evaluations)

			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
		
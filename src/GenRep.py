import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gym
from tqdm import tqdm
import random
from torch.autograd import Variable
from torch.distributions.normal import Normal
import random
import math

# Hyper parameters
BATCH_SIZE = 2048
EPOCHS = 30
BUFFER_SIZE = BATCH_SIZE
TRAIN_TO_TEST = 1
C = 0
SEED = 10

Series_Length = 7

g_input_size = 7   
g_hidden_size = 10
g_output_size = Series_Length

d_input_size = Series_Length
d_hidden_size = 10  
d_output_size = 1

# Make CPU if cuda doesnt work
device = "cuda"  

print_interval = 1000

torch.manual_seed(SEED)
np.random.seed(SEED)


class GenerativeReplay:
	def __init__(self):
		self.G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)#.to(device)
		self.D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)#.to(device)
		self.optGen = optim.Adam(self.G.parameters(), lr=2e-4)
		self.optDis = optim.Adam(self.D.parameters(), lr=3e-3)
		self.buffer = [None for x in range(int(BUFFER_SIZE))]
		self.optim_buffer = [None for x in range(int(BUFFER_SIZE))]
		self.highest_reward_state = None
		self.training = False
		self.use_optim_buffer = False
		self.i = 0
		self.sample_s = int(len(self.buffer)*TRAIN_TO_TEST)
		torch.set_printoptions(precision=3, sci_mode=False, linewidth=240, profile=None)

	def reset(self):
		del self.model
		self.__init__()

	def get_real_sample(self, sample_size, use_optim_buffer=False):
		res = None
		if not use_optim_buffer:
			res = torch.FloatTensor(random.sample(self.buffer, sample_size))#.to(device)
		else:
			res = torch.FloatTensor(random.sample(self.optim_buffer, sample_size))#.to(device)
		return res


	def get_noise_sample(self, sample_size):
		res = torch.rand(sample_size, Series_Length)#.to(device)
		return res

	def get_random_sample_from_buffer(self, sample_size):
		dat = torch.FloatTensor(self.buffer)
		dat_id = np.random.choice(dat.shape[0],
                                  size=sample_size,
                                  replace=False)
		return torch.FloatTensor(dat[dat_id, :])

	# Add new experiences as they come
	def add(self, state, action, next_state, reward, done):
		experience = [s for s in state]
		experience.append(action)
		experience.extend([s for s in next_state])
		experience.extend([reward, done])

		self.buffer[self.i] = experience
		self.i += 1

		if self.i >= BUFFER_SIZE:
			self.i = 0
			self.train()
			return True
		return False

	def edit_optim_buffer(self, reward):
		self.highest_reward_state = reward
		self.optim_buffer = self.buffer


	def cal_loss(self, decision, minibatch_size):
		criterion = nn.BCELoss()
		return criterion(decision, minibatch_size)


	# Train the model with what we have in the buffer and some generated data
	def train(self):
		train_data = None
		if self.optim_buffer and self.optim_buffer[0] != None:
			train_data = torch.FloatTensor(self.optim_buffer[:self.sample_s])
		else:
			train_data = torch.FloatTensor(self.buffer[:self.sample_s])

		# train_data = torch.FloatTensor(self.buffer[:self.sample_s])
		# train_data = self.get_random_sample_from_buffer(int(self.sample_s))

		for epoch in range(EPOCHS):
			self.D.zero_grad()
			# train D on actual data
			real_data = train_data #self.get_real_sample(sample_s)
			decision = self.D(real_data)
			error = self.cal_loss(decision, torch.ones(self.sample_s, 1 ))  # ones = true
			error.backward()    

			# train D on generated data
			noise = self.get_noise_sample(self.sample_s)
			fake_data = self.G(noise) 
			decision1 = self.D(fake_data)
			error1 = self.cal_loss( decision1, torch.zeros(self.sample_s,1))  # zeros = fake
			error1.backward()

			self.optDis.step()
		
			self.G.zero_grad()

			# train G
			noise1 = self.get_noise_sample(self.sample_s)
			fake_data1 = self.G(noise1)
			fake_decision = self.D(fake_data1)
			error2 = self.cal_loss(fake_decision, torch.ones(self.sample_s,1))  # we want to fool, so pretend it's all genuine
			error2.backward()

			loss,generated = error2.item(), fake_data1
			self.optGen.step()
			

	


	def sample(self, batch_size):	
		with torch.no_grad():
			noise_data = self.get_noise_sample(batch_size)
			outputs = self.G(noise_data)
			return (
					torch.FloatTensor(outputs[:, 0:2]),
					torch.FloatTensor(outputs[:, 2:3]),
					torch.FloatTensor(outputs[:, 3:5]),
					torch.FloatTensor(outputs[:, 5:6]),
					torch.FloatTensor(outputs[:, 6:7])
					)

	# def sample(self, batch_size):	
	# 	with torch.no_grad():
	# 		# 
	# 		noise_data = self.get_random_sample_from_buffer(batch_size)
	# 		outputs = self.G(noise_data)
	# 		state = noise_data[:, 0:2]
	# 		action = outputs[:, 2:3]
	# 		# print(state, action)
	# 		next_states, rewards, dones = self.calculate_next_states_rewards(state, action)
	# 		# print(len(state[0]),len(action[0]),len(next_states[0]),len(rewards[0]),len(dones[0]))

	# 		return (
	# 				torch.FloatTensor(state),
	# 				torch.FloatTensor(action),
	# 				torch.FloatTensor(next_states),
	# 				torch.FloatTensor(rewards),
	# 				torch.FloatTensor(dones)
	# 				)

	# def sample(self, batch_size):
	# 	with torch.no_grad():
	# 		#
	# 		noise_data = None
	# 		outputs = []
	# 		for i in range(batch_size):
	# 			noise_data = self.get_noise_sample(1)
	# 			outputs += self.G(noise_data).numpy().tolist()
	# 		outputs = torch.FloatTensor(outputs)
	# 		return (
	# 				torch.FloatTensor(outputs[:, 0:2]),
	# 				torch.FloatTensor(outputs[:, 2:3]),
	# 				torch.FloatTensor(outputs[:, 3:5]),
	# 				torch.FloatTensor(outputs[:, 5:6]),
	# 				torch.FloatTensor(outputs[:, 6:7])
	# 				)

	def calculate_next_states_rewards(self, states, actions):
		next_states = []
		rewards = []
		dones = []
		states_list = states.numpy().tolist()
		actions_list = actions.numpy().tolist()
		for i in range(len(states_list)):
			state = states_list[i]
			action = actions_list[i]

			position = state[0]
			velocity = state[1]
			force = min(max(action[0], -1.0), 1.0)
			velocity += force * 0.0015 - 0.0025 * math.cos(3 * position)
			if (velocity > 0.07): velocity = 0.07
			if (velocity < -0.07): velocity = -0.07
			position += velocity
			if (position > 0.6): position = 0.6
			if (position < -1.2): position = -1.2
			if (position == -1.2 and velocity < 0): velocity = 0
			next_states += [[position, velocity]]

			done = bool(position >= 0.45)
			if done:
				dones += [[1.0]]
			else:
				dones += [[0.0]]
			reward = 0
			if done:
				reward = 100.0
			reward -= math.pow(action[0], 2) * 0.1
			rewards += [[reward]]
		return next_states, rewards, dones

class Generator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Generator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size)
		self.map2 = nn.Linear(hidden_size, hidden_size)
		self.map3 = nn.Linear(hidden_size, output_size)
		self.xfer = torch.nn.SELU()

	def forward(self, x):
		x = self.xfer( self.map1(x) )
		x = self.xfer( self.map2(x) )
		return self.xfer( self.map3( x ) )


class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Discriminator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size)
		self.map2 = nn.Linear(hidden_size, hidden_size)
		self.map3 = nn.Linear(hidden_size, output_size)
		self.elu = torch.nn.ELU()

	def forward(self, x):
		x = self.elu(self.map1(x))
		x = self.elu(self.map2(x))
		return torch.sigmoid( self.map3(x) )
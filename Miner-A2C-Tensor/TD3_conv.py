import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from random import random, randrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, dropout= 0.4):
		super(Actor, self).__init__()
		self.conv1 = nn.Conv2d(state_dim[0], 32, 5, stride = 2, padding = 1)
		self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=1)
		self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.maxp4 = nn.MaxPool2d(2, 2)

		fc_input_dims = self.calculate_conv_output_dims(state_dim)
		self.l1 = nn.Linear(fc_input_dims, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.max_action = max_action
		# init weight and bias
		


	def forward(self, state):
		conv = F.leaky_relu(self.conv1(state))
		conv = F.leaky_relu(self.conv3(conv))
		conv = F.leaky_relu(self.maxp4(self.conv4(conv)))
		flatten = conv.view(conv.size()[0], -1)
		 
		a = F.leaky_relu(self.l1(flatten))
		#return self.max_action * torch.tanh(self.l3(a))
		#distribution = Categorical(F.softmax(self.l3(a), dim=-1))
		return self.l3(a)
	def calculate_conv_output_dims(self, input_dims):
		state = torch.zeros(1, *input_dims)
		dims = self.conv1(state)
		dims = self.conv3(dims)
		dims = self.conv4(dims)
		dims = self.maxp4(dims)
		return int(np.prod(dims.size()))

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, dropout = 0.4):
		super(Critic, self).__init__()

		# Q1 architecture
		self.conv1_1 = nn.Conv2d(state_dim[0], 32, 5, stride = 2, padding = 1)
		self.conv3_1 = nn.Conv2d(32, 64, 5, stride=2, padding=1)
		self.conv4_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.maxp4_1 = nn.MaxPool2d(2, 2)

		fc_input_dims = self.calculate_conv_output_dims(state_dim)
		self.l1 = nn.Linear(fc_input_dims + action_dim, 256)
		self.l3 = nn.Linear(256, 1)
		
		# Q2 architecture
		self.conv1_2 = nn.Conv2d(state_dim[0], 32, 5, stride = 2, padding = 1)
		self.conv3_2 = nn.Conv2d(32, 64, 5, stride=2, padding=1)
		self.conv4_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.maxp4_2 = nn.MaxPool2d(2, 2)
		#fc_input_dims = self.calculate_conv_output_dims(state_dim)

		self.l4 = nn.Linear(fc_input_dims + action_dim, 256)
		self.l6 = nn.Linear(256, 1)

		# init weights and biases layer


	def forward(self, state, action):
		conv1 = F.leaky_relu(self.conv1_1(state))
		conv1 = F.leaky_relu(self.conv3_1(conv1))
		conv1 = F.leaky_relu(self.maxp4_1(self.conv4_1(conv1)))
		flatten1 = conv1.view(conv1.size()[0], -1)
		sa = torch.cat([flatten1, action], 1)

		q1 = F.leaky_relu(self.l1(sa))
		q1 = self.l3(q1)

		# Q2
		conv2 = F.leaky_relu(self.conv1_2(state))
		conv2 = F.leaky_relu(self.conv3_2(conv2))
		conv2 = F.leaky_relu(self.maxp4_2(self.conv4_2(conv2)))
		flatten2 = conv2.view(conv2.size()[0], -1)
		sb = torch.cat([flatten2, action], 1)

		q2 = F.leaky_relu(self.l4(sb))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		conv1 = F.leaky_relu(self.conv1_1(state))
		conv1 = F.leaky_relu(self.conv3_1(conv1))
		conv1 = F.leaky_relu(self.maxp4_1(self.conv4_1(conv1)))
		flatten1 = conv1.view(conv1.size()[0], -1)
		sa = torch.cat([flatten1, action], 1)

		q1 = F.leaky_relu(self.l1(sa))
		q1 = self.l3(q1)
		return q1
	def calculate_conv_output_dims(self, input_dims):
		state = torch.zeros(1, *input_dims)
		dims = self.conv1_1(state)
		dims = self.conv3_1(dims)
		dims = self.conv4_1(dims)
		dims = self.maxp4_1(dims)
		return int(np.prod(dims.size()))


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		epsilon = 1,
		epsilon_min = 0.01, #The minimum epsilon 
		epsilon_decay = 0.999,#The decay epislon for each update_epsilon time
		discount=0.99,
		policy_noise=0.2,
		noise_clip = 0.2,
		learning_rate = 3e-4,
		tau=0.125,
		policy_freq=2,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = learning_rate)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = learning_rate)

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.total_it = 0

		self.loss_actor = 0
		self.loss_critic = 0


	def select_action(self, observ):
		#Chua update epsilon nen lay luon a_max
		state = torch.tensor([observ],dtype=torch.float).to(device)
		a_probs = self.actor(state).cpu().data.numpy().flatten()
		if (random() < self.epsilon):
			a_chosen = randrange(self.action_dim)
		else:
			a_chosen = np.argmax(a_probs)
		return a_chosen, a_probs
		#return Categorical(self.actor(state)).sample().cpu().numpy()[0]
	def predict_action(self, observ):
		state = torch.tensor([observ],dtype=torch.float).to(device)
		#distribution_action = Categorical(self.actor(state)).probs.detach().numpy()
		#a_max = np.argmax(distribution_action[0])
		distribution_action = self.actor(state).cpu().data.numpy().flatten()
		a_max = np.argmax(distribution_action)
		return a_max, distribution_action
	def update_epsilon(self):
		self.epsilon =  self.epsilon*self.epsilon_decay
		self.epsilon =  max(self.epsilon_min, self.epsilon)

	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		#print(next_state)
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			#action = self.actor(state)
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(0, self.max_action)
			#next_action = self.select_action(next_state)
			#probs_next_action = next_action.probs.detach().numpy()

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q
		# Get current Q estimates

		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.loss_critic = critic_loss.cpu().detach().numpy()
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			self.loss_actor = actor_loss.cpu().detach().numpy()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic", map_location = device))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location = device))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor", map_location = device))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location = device))
		self.actor_target = copy.deepcopy(self.actor)
		print("Load model sucessfully!")
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

		self.l1 = nn.Linear(state_dim - 3, 256)
		self.l2 = nn.Linear(256, 256)
		

		self.l3 = nn.Linear(3, 8)
		self.max_action = max_action

		self.l5 = nn.Linear(256 + 8, action_dim)
		# init weight and bias
		


	def forward(self, state):
		input1 = state[:,:-3]
		input2 = state[:,-3:]
		a = F.relu(self.l1(input1))
		a = F.relu(self.l2(a))
		b = F.relu(self.l3(input2))
		c = torch.cat((a,b),  dim = 1)
		#return self.max_action * torch.tanh(self.l3(a))
		#distribution = Categorical(F.softmax(self.l3(a), dim=-1))
		return F.softmax(self.l5(c), dim =-1)

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, dropout = 0.4):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim -3 + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256 + 8, 1)
		
		# Q2 architecture
		self.l4 = nn.Linear(state_dim -3 + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256 + 8, 1)

		# add
		self.l7 = nn.Linear(3 + action_dim, 8)

		self.l9 = nn.Linear(3 + action_dim, 8)


	def forward(self, state, action):
		input1 = state[:,:-3]
		input2 = state[:,-3:]
		sa = torch.cat([input1, action], 1)
		sb = torch.cat([input2, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1_ = F.relu(self.l7(sb))
		q1 = torch.cat([q1, q1_], 1)
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2_ = F.relu(self.l9(sb))
		q2 = torch.cat([q2, q2_], 1)
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		input1 = state[:,:-3]
		input2 = state[:,-3:]
		sa = torch.cat([input1, action], 1)
		sb = torch.cat([input2, action], 1)


		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1_ = F.relu(self.l7(sb))
		q1 = torch.cat([q1, q1_], 1)
		q1 = self.l3(q1)
		return q1


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
		learning_rate = 3e-4,
		policy_noise=0.2,
		noise_clip = 0.2,
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


	def select_action(self, state):
		#Chua update epsilon nen lay luon a_max
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		a_probs = self.actor(state).cpu().data.numpy().flatten()
		a_chosen = Categorical(self.actor(state)).sample().cpu()
		# if (random() < self.epsilon):
		# 	a_chosen = randrange(self.action_dim)
		# else:
		# 	a_chosen = np.argmax(a_probs)
		return int(a_chosen.numpy()), a_probs
		#return Categorical(self.actor(state)).sample().cpu().numpy()[0]
	def predict_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
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
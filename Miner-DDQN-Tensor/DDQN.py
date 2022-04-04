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


class Model(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, dropout= 0.4):
		super(Model, self).__init__()
		self.conv1 = nn.Conv2d(state_dim[0], 32, 5, stride = 2, padding = 1)
		self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=1)
		#self.maxp2 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.maxp3 = nn.MaxPool2d(2, 2)
		fc_input_dims = self.calculate_conv_output_dims(state_dim)


		self.l1 = nn.Linear(fc_input_dims, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.max_action = max_action
		# init weight and bias
	def calculate_conv_output_dims(self, input_dims):
		state = torch.zeros(1, *input_dims)
		dims = self.conv1(state)
		#dims = self.maxp1(dims)
		dims = self.conv2(dims)
		#dims = self.maxp2(dims)
		dims = self.conv3(dims)
		dims = self.maxp3(dims)
		return int(np.prod(dims.size()))	


	def forward(self, state):
		#conv = F.relu(self.maxp1(self.conv1(state)))
		conv = F.leaky_relu(self.conv1(state))
		#conv = F.relu(self.maxp2(self.conv2(conv)))
		conv = F.leaky_relu(self.conv2(conv))
		conv = F.leaky_relu(self.maxp3(self.conv3(conv)))
		#conv = F.relu(self.conv3(conv))
		flatten = conv.view(conv.size()[0], -1)
		 
		a = F.leaky_relu(self.l1(flatten))
		a = F.leaky_relu(self.l2(a))
		#return self.max_action * torch.tanh(self.l3(a))
		#distribution = Categorical(F.softmax(self.l3(a), dim=-1))
		return self.l3(a)

class DDQN(object):
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
		tau=1e-3,
		policy_freq=2,
		replace = 1000
	):

		self.actor = Model(state_dim, action_dim, max_action).to(device)
		#self.actor_target = Model(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr = learning_rate)

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.total_it = 0
		self.replace_target_cnt = replace
		self.learn_step_counter = 0

		self.loss_value = 0


	def select_action(self, observ):
		#Chua update epsilon nen lay luon a_max
		state = torch.tensor([observ],dtype=torch.float).to(device)
		a_probs = self.actor(state).cpu().data.numpy().flatten()
		if (random() < self.epsilon):
			a_chosen = randrange(self.action_dim)
		else:
			a_chosen = np.argmax(a_probs)
		return a_chosen
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
	def replace_target_network(self):
		if self.replace_target_cnt is not None and \
			self.learn_step_counter % self.replace_target_cnt == 0:
			self.actor_target.load_state_dict(self.actor.state_dict())

	def train(self, replay_buffer, batch_size=100):
		self.actor_optimizer.zero_grad()

		self.replace_target_network()

		# histories, actions, rewards, next_histories, dones = self.sample_tensor(replay_buffer, batch_size)
		histories, actions, next_histories, rewards, not_dones = replay_buffer.sample(batch_size)

		indices = np.arange(batch_size)
		actions = actions.detach().reshape(-1,)
		q_pred = self.actor.forward(histories)[indices, actions].reshape((batch_size, -1))
		with torch.no_grad():
			actor_target = self.actor_target.forward(next_histories)
			actor_eval = self.actor.forward(next_histories)
			max_actions = torch.argmax(actor_eval, dim=1)
			#actor_target[dones] = 0.0
			q_target = rewards + self.discount*actor_target[indices, max_actions].reshape((batch_size, -1))*not_dones
		loss = F.mse_loss(q_pred, q_target).to(device)
		self.loss_value = loss.cpu().detach().numpy()
		loss.backward()

		self.actor_optimizer.step()
		self.learn_step_counter += 1

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):

		self.actor.load_state_dict(torch.load(filename + "_actor", map_location = device))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location = device))
		self.actor_target = copy.deepcopy(self.actor)
		print("Load model sucessfully!")
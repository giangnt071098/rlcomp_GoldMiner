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
		self.conv1 = nn.Conv2d(state_dim[0], 32, 7, stride = 2, padding = 1)
		#self.maxp1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=1)
		#self.maxp2 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		#self.maxp3 = nn.MaxPool2d(2, 2)
		fc_input_dims = self.calculate_conv_output_dims(state_dim)


		self.l1 = nn.Linear(fc_input_dims, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, action_dim)
		self.max_action = max_action
		# init weight and bias
	def calculate_conv_output_dims(self, input_dims):
		state = torch.zeros(1, *input_dims)
		dims = self.conv1(state)
		#print(dims.size())
		#dims = self.maxp1(dims)
		dims = self.conv2(dims)
		#print(dims.size())
		#dims = self.maxp2(dims)
		dims = self.conv3(dims)
		#dims = self.maxp3(dims)
		#print(dims.size())
		return int(np.prod(dims.size()))	


	def forward(self, state):
		#conv = F.leaky_relu(self.maxp1(self.conv1(state)))
		conv = F.leaky_relu(self.conv1(state))
		#conv = F.leaky_relu(self.maxp2(self.conv2(conv)))
		conv = F.leaky_relu(self.conv2(conv))
		#conv = F.leaky_relu(self.maxp3(self.conv3(conv)))
		conv = F.leaky_relu(self.conv3(conv))
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
		epsilon_min = 0.05, #The minimum epsilon 
		epsilon_decay = 0.9995,#The decay epislon for each update_epsilon time
		discount=0.99,
		learning_rate = 3e-4,
		replace = 1000
	):

		self.model1 = Model(state_dim, action_dim, max_action).to(device)
		self.model2 = Model(state_dim, action_dim, max_action).to(device)
		#self.actor_target = copy.deepcopy(self.actor)
		self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr = learning_rate)
		self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr = learning_rate)

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min

		self.loss_value = 0
		self.state_dim = state_dim


	def select_action(self, observ, bot_action = None):
		#Chua update epsilon nen lay luon a_max
		#state = torch.tensor([observ],dtype=torch.float).to(device)
		state = torch.reshape(observ, (-1,*self.state_dim)).to(device)
		a_model1 = self.model1(state).cpu().data.numpy().flatten()
		a_model2 = self.model2(state).cpu().data.numpy().flatten()
		if np.max(a_model1) > np.max(a_model2):
			a_max = np.argmax(a_model1)
		else: a_max = np.argmax(a_model2)
		if bot_action == None:
			if (random() < self.epsilon):
				a_chosen = randrange(self.action_dim)
			else:
				a_chosen = a_max
		else:
			if (random() < self.epsilon):
				a_random = randrange(self.action_dim)
				a_chosen = np.random.choice((bot_action, a_random), 1, p=[0.3, 0.7])[0]
			else:
				a_chosen = a_max
		return a_chosen
		#return Categorical(self.actor(state)).sample().cpu().numpy()[0]
	def predict_action(self, observ):
		#state = torch.tensor([observ],dtype=torch.float).to(device)
		state = torch.reshape(observ, (-1,*self.state_dim)).to(device)
		a_model1 = self.model1(state).cpu().data.numpy().flatten()
		a_model2 = self.model2(state).cpu().data.numpy().flatten()
		if np.max(a_model1) > np.max(a_model2):
			a_chosen = np.argmax(a_model1)
		else: a_chosen = np.argmax(a_model2)
		return a_chosen, a_model1
	def update_epsilon(self):
		self.epsilon =  self.epsilon*self.epsilon_decay
		self.epsilon =  max(self.epsilon_min, self.epsilon)

	def train(self, replay_buffer, batch_size=100):


		# histories, actions, rewards, next_histories, dones = self.sample_tensor(replay_buffer, batch_size)
		histories, actions, next_histories, rewards, not_dones = replay_buffer.sample(batch_size)

		indices = np.arange(batch_size)
		actions = actions.view(actions.size(0), 1)

		# current Q
		curr_Q1 = self.model1.forward(histories).gather(1, actions)
		curr_Q2 = self.model2.forward(histories).gather(1, actions)

		next_Q1 = self.model1.forward(next_histories)
		next_Q2 = self.model2.forward(next_histories)
		# print(next_Q1, next_Q2)
		# print(torch.max(next_Q1, 1)[0], torch.max(next_Q2, 1)[0])
		next_Q = torch.min(torch.max(next_Q1, 1)[0], torch.max(next_Q2, 1)[0])
		next_Q = next_Q.view(next_Q.size(0), 1)
		with torch.no_grad():	
			expected_Q = rewards + self.discount*next_Q*not_dones
		loss1 = F.mse_loss(curr_Q1, expected_Q).to(device)
		loss2 = F.mse_loss(curr_Q2, expected_Q).to(device)
		self.loss_value = loss1.cpu().detach().numpy()


		# backward
		self.optimizer1.zero_grad()
		loss1.backward()
		self.optimizer1.step()

		self.optimizer2.zero_grad()
		loss2.backward()
		self.optimizer2.step()


	def save(self, filename):
		
		torch.save(self.model1.state_dict(), filename + "_model1")
		torch.save(self.optimizer1.state_dict(), filename + "_model1_optimizer")
		torch.save(self.model2.state_dict(), filename + "_model2")
		torch.save(self.optimizer2.state_dict(), filename + "_model2_optimizer")


	def load(self, filename):

		self.model1.load_state_dict(torch.load(filename + "_model1", map_location = device))
		self.optimizer1.load_state_dict(torch.load(filename + "_model1_optimizer", map_location = device))
		self.model2.load_state_dict(torch.load(filename + "_model2", map_location = device))
		self.optimizer2.load_state_dict(torch.load(filename + "_model2_optimizer", map_location = device))
		print("Load model sucessfully!")
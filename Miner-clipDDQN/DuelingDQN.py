import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from random import random, randrange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Model(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, dropout= 0.4):
		super(Model, self).__init__()
		self.state_dim = state_dim

		self.conv_feature = nn.Sequential(
			nn.Conv2d(state_dim[0], 32, 8, stride = 4, padding = 1),
			nn.LeakyReLU(),
			nn.Conv2d(32, 64, 5, stride = 2, padding = 1),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
			#nn.MaxPool2d((2,2)),
			nn.LeakyReLU()
		)
		fc_input_dims = self.feature_size()
		self.value_stream = nn.Sequential(
			nn.Linear(fc_input_dims, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 1)
		)

		self.advantage_stream = nn.Sequential(
			nn.Linear(fc_input_dims, 256),
			nn.LeakyReLU(),
			nn.Linear(256, action_dim)
		)
		self.max_action = max_action
		# init weight and bias
	def feature_size(self):	
		# state = torch.zeros(1, *self.state_dim)
		# dims = self.conv_feature[2](dims)
		# dims = self.conv_feature[4](dims)
		return self.conv_feature(autograd.Variable(torch.zeros(1, *self.state_dim))).view(1, -1).size(1)	


	def forward(self, state):
		conv_ = self.conv_feature(state)
		conv_ = conv_.view(conv_.size(0), -1)
		values = self.value_stream(conv_)
		advantage = self.advantage_stream(conv_)

		qvals = values + (advantage - advantage.mean())
		return qvals

class DuelingDQN(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		epsilon = 1,
		epsilon_min = 0.05, #The minimum epsilon 
		epsilon_decay = 0.99999,#The decay epislon for each update_epsilon time
		eps_dec=1e-5,
		discount=0.99,
		learning_rate = 3e-4,
		replace = 1000
	):

		self.model = Model(state_dim, action_dim, max_action).to(device)

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.eps_dec = eps_dec

		#self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = learning_rate, momentum =0.95)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
		self.MSE_loss = nn.MSELoss()

		self.loss_value = 0
		self.state_dim = state_dim

	def select_action(self, observ, bot_action = None):
		# #Chua update epsilon nen lay luon a_max
		# state = torch.tensor([observ],dtype=torch.float).to(device)
		# a_model = self.model.forward(state).detach().numpy()
		# if (random() < self.epsilon):
		# 	a_chosen = randrange(self.action_dim)
		# else:
		# 	a_chosen = np.argmax(a_model)
		# return a_chosen
		# #return Categorical(self.actor(state)).sample().cpu().numpy()[0]

		#state = torch.tensor([observ],dtype=torch.float).to(device)
		state = torch.reshape(observ, (-1,*self.state_dim)).to(device)
		a_model = self.model.forward(state).cpu().data.numpy().flatten()
		a_max = np.argmax(a_model)
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
	def predict_action(self, observ):
		#state = torch.tensor([observ],dtype=torch.float).to(device)
		state = torch.reshape(observ, (-1,*self.state_dim)).to(device)
		a_model = self.model(state).cpu().data.numpy().flatten()
		a_chosen = np.argmax(a_model)
		return a_chosen, a_model
	def update_epsilon(self):
		self.epsilon =  self.epsilon*self.epsilon_decay
		self.epsilon =  max(self.epsilon_min, self.epsilon)
	def decre_epsilon(self):
		self.epsilon = (self.epsilon - self.eps_dec) \
                           if self.epsilon > self.epsilon_min else self.epsilon_min

	def train(self, replay_buffer, batch_size=100):


		# histories, actions, rewards, next_histories, dones = self.sample_tensor(replay_buffer, batch_size)
		histories, actions, next_histories, rewards, not_dones = replay_buffer.sample(batch_size)

		indices = np.arange(batch_size)
		actions = actions.view(actions.size(0), 1)

		# current Q
		curr_Q = self.model.forward(histories).gather(1, actions)
		#curr_Q = curr_Q.squeeze(1)


		next_Q = self.model.forward(next_histories)
		max_next_Q = torch.max(next_Q, 1)[0].unsqueeze(1)


		with torch.no_grad():
			expected_Q = rewards + self.discount * max_next_Q * not_dones

		loss = self.MSE_loss(curr_Q, expected_Q).to(device)
		self.loss_value = loss.cpu().detach().numpy()


		# backward
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		#self.decre_epsilon()


	def save(self, filename):
		
		torch.save(self.model.state_dict(), filename + "_model")
		torch.save(self.optimizer.state_dict(), filename + "_model_optimizer")


	def load(self, filename):

		self.model.load_state_dict(torch.load(filename + "_model", map_location = device))
		self.optimizer.load_state_dict(torch.load(filename + "_model_optimizer", map_location = device))
		print("Load model sucessfully!")
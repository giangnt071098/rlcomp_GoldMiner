import copy 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from random import random, randrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, dropout= 0.4):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.max_action = max_action
		# init weight and bias
		self._initialize_weights()


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		#return self.max_action * torch.tanh(self.l3(a))
		#distribution = Categorical(F.softmax(self.l3(a), dim=-1))
		return F.softmax(self.l3(a), dim =-1)
	def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, dropout = 0.4):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)
		# init weights and biases layer
		self._initialize_weights()

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


	def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
class PPO(nn.Module):
	def __init__(self, 
				state_dim,
				action_dim,
				max_action,
				epsilon=1,
				epsilon_min = 0.01,
				epsilon_decay = 0.999,
				discount=0.99,
				learning_rate = 3e-4,
				tau=0.125,):
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

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.total_it = 0

		self.loss_actor = 0
		self.loss_critic = 0
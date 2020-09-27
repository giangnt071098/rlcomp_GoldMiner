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

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		self.max_action = max_action
		# init weight and bias
		


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = self.l3(a)
		#return self.max_action * torch.tanh(self.l3(a))
		distribution = Categorical(F.softmax(a, dim=-1))
		return distribution

class Critic(nn.Module):
	def __init__(self, state_dim, dropout = 0.4):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class A2C(object):
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
        tau=0.125,
        policy_freq=2,
        ):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = learning_rate)
        
        self.critic = Critic(state_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = learning_rate)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.total_it = 0

        self.loss_actor = 0
        self.loss_critic = 0
        
        
    def select_action(self, state):
		#Chua update epsilon nen lay luon a_max
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		# a_max = np.argmax(self.actor(state).cpu().data.numpy().flatten())
		# if (random() < self.epsilon):
		# 	a_chosen = randrange(self.action_dim)
		# else:
		# 	a_chosen = a_max
		# return a_chosen
        return self.actor(state).sample().item()
    def predict_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		#distribution_action = Categorical(self.actor(state)).probs.detach().numpy()
		#a_max = np.argmax(distribution_action[0])
        distribution_action = self.actor(state)
        a_max = np.argmax(distribution_action)
        return a_max, distribution_action
    def update_epsilon(self):
        self.epsilon =  self.epsilon*self.epsilon_decay
        self.epsilon =  max(self.epsilon_min, self.epsilon)
        
    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        probs = self.actor(state)
        print(probs)
        print(action)
        print(probs[action])
        # Get current Q estimates
        current_Q = self.critic(state)
        with torch.no_grad():
			# Select action according to policy and add clipped noise
			
			#next_action = self.actor_target(next_state).sample()
			#next_action = self.select_action(next_state)
			#probs_next_action = next_action.probs.detach().numpy()
			# Compute the target Q value
            target_Q = self.critic_target(next_state)

        advantage = reward + not_done * self.discount * target_Q - current_Q
		
		

		# Compute critic loss
        critic_loss = advantage.pow(2).mean()
        self.loss_critic = critic_loss.detach().numpy()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()



		# Compute actor losse
        actor_loss = -1
        self.loss_actor = actor_loss.detach().numpy()
		
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
		
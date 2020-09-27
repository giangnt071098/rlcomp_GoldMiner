import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim = 1, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state = np.zeros((max_size, *state_dim), dtype=np.float32)
		
		self.action = np.zeros((max_size, action_dim), dtype=np.int64)
		self.next_state = np.zeros((max_size, *state_dim), dtype=np.float32)
		self.reward = np.zeros((max_size, 1), dtype=np.float32)
		self.not_done = np.zeros((max_size, 1), dtype=np.float32)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1.0 - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.tensor(self.state[ind]).to(self.device),
			torch.tensor(self.action[ind]).to(self.device),
			torch.tensor(self.next_state[ind]).to(self.device),
			torch.tensor(self.reward[ind]).to(self.device),
			torch.tensor(self.not_done[ind]).to(self.device)
		)
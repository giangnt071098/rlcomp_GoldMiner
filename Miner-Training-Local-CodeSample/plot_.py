import numpy as np
import matplotlib.pyplot as plt
plt_before, plt_value_before = 0, 0
episode_i, total_reward = 0, 0
while(1):
	f = open("data.txt", "r")
	data = f.readlines()
	f.close()
	if len(data) >0:
		value = [int(s) for s in data[0][1:-1].split(', ')]
		total_reward, episode_i = value[0], value[1]
		plt.plot([plt_before, episode_i],[plt_value_before, total_reward], '-ro', marker = 'None')
		plt.pause(0.1)
		plt_before, plt_value_before = episode_i, total_reward
    

plt.show()
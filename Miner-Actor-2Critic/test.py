import numpy as np
from MinerEnv import MinerEnv
from random import random, randrange
import torch
import random
import os
import json
import torch.nn.functional as F
# HOST = 'localhost'
# PORT = 1111
# minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
# minerEnv.start()  # Connect to the game
# mapID = 1
# posID_x = 1
# posID_y = 0
# #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
# request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
# #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
# minerEnv.send_map_info(request)
# minerEnv.reset()
# s = minerEnv.get_state()
# action = 0
# print(minerEnv.state.score, minerEnv.state.x, minerEnv.state.y)
# minerEnv.step(str(action))
# action = 5
# minerEnv.step(str(action))
# reward = minerEnv.get_reward()
# s_next = minerEnv.get_state()
# print(minerEnv.state.score, minerEnv.state.x, minerEnv.state.y, minerEnv.state.score, reward)



map = []
for i in range(1,7):
	filename = "map"+str(i)
	print("Found: " + filename)
	with open(os.path.join("Maps", filename), 'r') as f:
		map.append(f.read())
a = np.zeros((9,21), dtype=np.int32)
b = {}
for i in range(6):
	b[i] = json.loads(map[i])
for i in range(9):
	for j in range(21):
		a[i][j] = int(max(b[0][i][j],b[1][i][j],b[2][i][j],b[3][i][j],b[4][i][j]))
		print(a[i][j])
with open(os.path.join("Maps", "map8"), 'a') as f:
	for i in range(9):
		f.write('[')
		for j in range(21):
			f.write(str(a[i][j]))
			f.write(',')
		f.write('],')


import sys
import numpy as np
from GAME_SOCKET import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
from random import random, randrange

TreeID = 1
TrapID = 2
SwampID = 3
class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score#Storing the last score for designing the reward function

    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, request):#tell server which map to run
        self.socket.send(request)

    def reset(self): #start new game
        self.state_x_pre = self.state.x
        self.state_y_pre = self.state.y
        self.last3position = []
        self.Swamp_position = []
        try:
            message = self.socket.receive() #receive game info from server
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.state_x_pre = self.state.x
        self.state_y_pre = self.state.y
        self.last3position.append([self.state.x, self.state.y])
        if len(self.last3position) > 3:
            self.last3position.pop(0)
        #print(self.last3position)
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            self.state.update_state(message) #update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
                    view[i, j] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = self.state.mapInfo.gold_amount(i, j)

        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        
        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)
        #Add position of bots 
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                DQNState.append(player["posx"])
                DQNState.append(player["posy"])
                
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)
        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        if score_action > 0:
            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action
            
        #If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= TreeID*3* randrange(1, 5)
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= TrapID*3
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            if [self.state.x, self.state.y] in self.Swamp_position: # go to Swamp again                
                reward -= 15
            else:
                reward -= SwampID*3 # first time go to swamp
            self.Swamp_position.append([self.state.x, self.state.y])
        if self.state.mapInfo.gold_amount(self.state.x, self.state.y) >=50:
            reward += self.state.mapInfo.gold_amount(self.state.x, self.state.y)/800
        if self.state.mapInfo.gold_amount(self.state_x_pre, self.state_y_pre) >= 50 and self.state.lastAction != 5: # in gold but don't craft
            reward -= 10
        if self.state.lastAction == 5 and score_action < 0: # not in gold but craft
            reward -= 10
        if len(self.last3position) == 3 and self.state.lastAction !=5: # back to same position
            if self.last3position[0] == self.last3position[2]:
                reward -= 4
            if self.last3position[1] == self.last3position[2]:
                reward -= 4
        if self.state.energy >= 45 and self.state.lastAction == 4:
            reward -= 7
        # if self.state.status == State.STATUS_PLAYING:
        #     reward += 0.5
        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -40
            
        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -22
        # print ("reward",reward)
        return reward
    def get_state2(self, limit):
        # Building the map
        view = np.zeros([limit*2+1, limit*2+1], dtype=int)
        max_x, max_y = self.state.mapInfo.max_x, self.state.mapInfo.max_y
        xlimit_below = np.clip(self.state.x - limit, 0, max_x) - np.clip(self.state.x + limit - max_x, 0, limit)
        xlimit_up = np.clip(self.state.x + limit, 0, max_x) + np.clip(0 - self.state.x + limit, 0, limit)
        ylimit_below = np.clip(self.state.y - limit, 0, max_y) - np.clip(self.state.y + limit - max_y, 0, limit)
        ylimit_up = np.clip(self.state.y + limit, 0, max_y) + np.clip(0 - self.state.y + limit, 0, limit)

        #print(xlimit_below, xlimit_up, ylimit_below, ylimit_up, self.state.x, self.state.y)

        for i in range(xlimit_below, xlimit_up + 1):
            for j in range(ylimit_below, ylimit_up + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i - xlimit_below, j - ylimit_below] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i - xlimit_below, j - ylimit_below] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
                    view[i - xlimit_below, j - ylimit_below] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i - xlimit_below, j - ylimit_below] = self.state.mapInfo.gold_amount(i, j)/10
        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        
        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x - xlimit_below)
        DQNState.append(self.state.y - ylimit_below)
        DQNState.append(self.state.energy)
        #Add position of bots 
        # for player in self.state.players:
        #     if player["playerId"] != self.state.id:
        #         DQNState.append(player["posx"])
        #         DQNState.append(player["posy"])
                
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)
        return DQNState
    def get_state3(self, limit):
        # Building the map
        view = np.zeros([limit*2+1, limit*2+1], dtype=int)
        max_x, max_y = self.state.mapInfo.max_x, self.state.mapInfo.max_y
        xlimit_below = np.clip(self.state.x - limit, 0, max_x) - np.clip(self.state.x + limit - max_x, 0, limit)
        xlimit_up = np.clip(self.state.x + limit, 0, max_x) + np.clip(0 - self.state.x + limit, 0, limit)
        ylimit_below = np.clip(self.state.y - limit, 0, max_y) - np.clip(self.state.y + limit - max_y, 0, limit)
        ylimit_up = np.clip(self.state.y + limit, 0, max_y) + np.clip(0 - self.state.y + limit, 0, limit)

        #print(xlimit_below, xlimit_up, ylimit_below, ylimit_up, self.state.x, self.state.y)
        dmax, m, n, exist_gold = -1000, -5, 0.1, False
        x_maxgold, y_maxgold = self.state.x, self.state.y
        for i in range(max_x + 1):
            for j in range(max_y + 1):
                if self.state.mapInfo.gold_amount(i, j) >=50:
                    exist_gold = True
                    d = m*((self.state.x - i)**2 + (self.state.y - j)**2) + n*self.state.mapInfo.gold_amount(i, j)
                    if d > dmax:
                        dmax = d
                        x_maxgold, y_maxgold = i, j # position of cell is nearest and much gold

                if i in range(xlimit_below, xlimit_up + 1) and j in range(ylimit_below, ylimit_up + 1):
                    if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                        view[i - xlimit_below, j - ylimit_below] = -TreeID
                    if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                        view[i - xlimit_below, j - ylimit_below] = -TrapID
                    if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
                        view[i - xlimit_below, j - ylimit_below] = -SwampID
                    if self.state.mapInfo.gold_amount(i, j) > 0:
                        view[i - xlimit_below, j - ylimit_below] = self.state.mapInfo.gold_amount(i, j)/10
        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        
        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x - xlimit_below)
        DQNState.append(self.state.y - ylimit_below)
        DQNState.append(self.state.energy)
        #Add position of bots 
        # for player in self.state.players:
        #     if player["playerId"] != self.state.id:
        #         DQNState.append(player["posx"])
        #         DQNState.append(player["posy"])
        DQNState.append(self.state.x - x_maxgold)
        DQNState.append(self.state.y - y_maxgold)
        if exist_gold == False:
            DQNState.append(0)
        else: DQNState.append(self.state.mapInfo.gold_amount(x_maxgold, y_maxgold)/10)      
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)
        return DQNState
    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING

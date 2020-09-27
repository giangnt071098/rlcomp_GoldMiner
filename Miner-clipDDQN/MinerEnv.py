import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
from random import random, randrange
import torch

TreeID = 1
TrapID = 2
SwampID = 3
LandID = 0
def distance(posA, posB):
    return abs(posA[0] - posB[0]) + abs(posA[1] - posB[1])
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
        self.craft_no_gold = 0
        self.in_gold = 0
        self.premindist = 1000
        #self.dmax,_,_ = self.distance_value_trade_off()
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
    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        # dmax, xgold, ygold = self.distance_value_trade_off()
        # print("come to: ", dmax, xgold, ygold, self.state.mapInfo.gold_amount(xgold, ygold))
        # if dmax >= self.dmax:
        #     reward += 0.1
        # print(self.dmax, self.state.x, self.state.y, self.state.mapInfo.gold_amount(self.state.x, self.state.y))
        # self.dmax = dmax
        

        if score_action > 0:
            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action/50*10
            
        #If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= 0.06*3
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= 0.03*3
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            if [self.state.x, self.state.y] in self.Swamp_position: # go to Swamp again                
                reward -= 0.5
            else:
                reward -= 0.05 # first time go to swamp
            #reward -=0.4
        if self.state.mapInfo.gold_amount(self.state.x, self.state.y) >=50:
            reward += 0.3
        if self.state.mapInfo.gold_amount(self.state_x_pre, self.state_y_pre) >= 50 and self.state.lastAction != 5: # in gold but don't craft
            self.in_gold += 1
            reward -= 0.5
        if self.state.lastAction == 5 and score_action == 0: # not in gold but craft
            self.craft_no_gold +=1
            reward -= 0.5
        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward -= 20
            
        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward -= 0.7   
        if self.state.status == State.STATUS_PLAYING:
            reward += 0.1
        # print ("reward",reward)
        return reward
    def get_reward_complex(self):
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        ### reward for gold
        golds = self.state.mapInfo.golds
        miner_posx, miner_posy = self.state.x, self.state.y
        target_x, target_y = miner_posx, miner_posy
        mindist = 1000
        for gold in golds:
            dist = distance([gold["posx"], gold["posy"]], [miner_posx, miner_posy]) - self.reward_gold([gold["posx"], gold["posy"]])
            if dist < mindist:
                mindist = dist
        if mindist < self.premindist:
            reward += 0.5
        self.premindist = mindist
        ####
        if score_action > 0:
            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action/50*10
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= 0.06*3
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= 0.03*3
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            if [self.state.x, self.state.y] in self.Swamp_position: # go to Swamp again                
                reward -= 0.5
            else:
                reward -= 0.05 # first time go to swamp
            #reward -=0.4
        if self.state.mapInfo.gold_amount(self.state_x_pre, self.state_y_pre) >= 50 and self.state.lastAction != 5: # in gold but don't craft
            reward -= 0.5
        if self.state.lastAction == 5 and score_action == 0: # not in gold but craft
            reward -= 0.5
        if self.state.energy >= 45 and self.state.lastAction == 4:
            reward -= 1
        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward -= 20
            
        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward -= 0.7   
        if self.state.status == State.STATUS_PLAYING:
            reward += 0.1
        return reward

    def reward_gold(self, gold_pos):
        x,y = gold_pos[0], gold_pos[1]
        reward = 0
        for stt, (i, j) in enumerate(zip([-1,1,0,0,-1,1,-1,1],[0,0,-1,1,-1,-1,1,1])):
            xnew, ynew = x+i, y+j
            if xnew <= self.state.mapInfo.max_x and xnew >=0 \
            and ynew <= self.state.mapInfo.max_y and ynew >= 0:
                amount = self.state.mapInfo.gold_amount(xnew, ynew)
                if  amount>=100 and amount<= 200:
                    reward += 1
                if amount >200 and amount <=500:
                    reward += 2
                if amount >500:
                    reward += 3
                if amount >=1000:
                    reward += 5
        return reward

    def get_state_tensor(self, scale_map):
        n = scale_map
        view = torch.zeros((7,n*(self.state.mapInfo.max_x + 1), n*(self.state.mapInfo.max_y + 1)), dtype =torch.float)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree     # trap map
                    view[2,n*i:n*i+n, n*j:n*j+n] = -TreeID
                    view[0,n*i:n*i+n, n*j:n*j+n] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap     # trap map
                    view[2,n*i:n*i+n, n*j:n*j+n] = -TrapID
                    view[0,n*i:n*i+n, n*j:n*j+n] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp    # trap map
                    if [i, j] not in self.Swamp_position:
                        view[2,n*i:n*i+n, n*j:n*j+n] = -SwampID # -3
                        view[0,n*i:n*i+n, n*j:n*j+n] = -SwampID
                    else:
                        view[2,n*i:n*i+n, n*j:n*j+n] = -SwampID - 3# -6
                        view[0,n*i:n*i+n, n*j:n*j+n] = -SwampID - 3
                gold_ = self.state.mapInfo.gold_amount(i, j)
                if gold_ > 0:
                    view[1,n*i:n*i+n, n*j:n*j+n] = gold_/1000 ##/10 gold map
                    view[0,n*i:n*i+n, n*j:n*j+n] = gold_/1000

        index = 3
        playerid_list = []
        for stt,player in enumerate(self.state.players):
            playerid_list.append(player["playerId"])
            if player["playerId"] != self.state.id:
                try:
                    if player["status"] not in [1,2,3]:
                        try:
                            view[index + 1, n*player["posx"]:n*player["posx"]+n,n*player["posy"]:n*player["posy"]+n] = player["energy"]/50
                        except:
                            view[index + 1, n*player["posx"]:n*player["posx"]+n,n*player["posy"]:n*player["posy"]+n] = 1
                        index += 1
                except:
                    view[index+1, n*player["posx"]: n*player["posx"]+n,n*player["posy"]:n*player["posy"]+n] = 1
                    # print(self.state.players)
                    #print(view[player["posx"]: player["posx"]+1, player["posy"]: player["posy"]+1, stt])
                    #print(np.unique(a-view[:,:,stt]))
                    index += 1
            else:
                try:
                    view[3, n*self.state.x:n*self.state.x+n,n*self.state.y:n*self.state.y+n] = self.state.energy/50
                except: 
                    print('out of map')
        if self.state.id not in playerid_list:
            view[3, n*self.state.x:n*self.state.x+n,n*self.state.y:n*self.state.y+n] = self.state.energy/50
        #print("check: ", np.unique(view[3,:,:]))      
        DQNState = view
        return DQNState
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
    def update_swamp(self):
        for player in self.state.players:
            if self.state.mapInfo.get_obstacle(player["posx"], player["posy"]) == 3 and [player["posx"], player["posy"]] not in self.Swamp_position:
                self.Swamp_position.append([player["posx"], player["posy"]])
    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import torch
from MinerEnv import MinerEnv
import numpy as np
import TD3, DDPG, newDDPG, A2C, newTD3, TD3_noise
from models_TD3_2 import TD3_bot
from models_newTD3_2 import newTD3_bot
from models_TD3_lrelu import TD3

ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5

MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map
limit = 2
HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])
state_dim = (2*limit+1)**2 + 3
action_dim = 6
max_action = 1.0
# load model
kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }
policy = TD3_bot.TD3(**kwargs)
policy_file = "TD3_Miner_0_2"
policy.load(f"./models_TD3_2/{policy_file}")
print("Loaded model from disk")
status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
                  3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
total_reward = 0
def distance(posA, posB):
    return abs(posA[0] - posB[0]) + abs(posA[1] - posB[1])
def no_craft_gold():
    #them tu day
    golds = minerEnv.state.mapInfo.golds
    mindis = 1000
    miner_posx, miner_posy = minerEnv.state.x, minerEnv.state.y
    for gold in golds:
        dist = distance([gold["posx"], gold["posy"]], [miner_posx, miner_posy])
        if  dist < mindis:
            mindis = dist
            target_x = gold["posx"]
            target_y = gold["posy"]
    for i in range(abs(target_x - miner_posx)):
        if minerEnv.state.energy <10: minerEnv.step(str(4))
        if (target_x - miner_posx) < 0:
            minerEnv.step(str(0))
        if (target_x - miner_posx) > 0:
            minerEnv.step(str(1))
    for j in range(abs(target_y - miner_posy)):
        if minerEnv.state.energy <10: minerEnv.step(str(4))
        if (target_y - miner_posy) < 0:
            minerEnv.step(str(2))
        if (target_y - miner_posy) > 0:
            minerEnv.step(str(3))
    minerEnv.step(str(5))

    #them den day
def no_craft_gold2():
    #them tu day
    golds = minerEnv.state.mapInfo.golds
    mindis = 1000
    miner_posx, miner_posy = minerEnv.state.x, minerEnv.state.y
    for gold in golds:
        dist = distance([gold["posx"], gold["posy"]], [miner_posx, miner_posy])
        if  dist < mindis:
            mindis = dist
            target_x = gold["posx"]
            target_y = gold["posy"]
    if (target_x - miner_posx) < 0:
        return 0
    if (target_x - miner_posx) > 0:
        return 1
    if (target_y - miner_posy) < 0:
        return 2
    if (target_y - miner_posy) > 0:
        return 3
    return 5
    #them den day
def check_gold():
    x,y = minerEnv.state.x, minerEnv.state.y
    max_gold, pre_action = 0, 4
    for stt, (i, j) in enumerate(zip([-1,1,0,0],[0,0,-1,1])):
        xnew, ynew = x+i, y+j
        if xnew <= minerEnv.state.mapInfo.max_x and xnew >=0 \
        and ynew <= minerEnv.state.mapInfo.max_y and ynew >= 0:
            if minerEnv.state.mapInfo.gold_amount(xnew, ynew) > max_gold:
                pre_action, max_gold = stt, minerEnv.state.mapInfo.gold_amount(xnew, ynew)
    return pre_action if pre_action != 4 else 10
def newPos(oldpos, action):
    action_dict = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1], 4: [0, 0], 5: [0, 0]}
    newpos = np.add(oldpos, action_dict[action])
    if newpos[0] > minerEnv.state.mapInfo.max_x or newpos[1] > minerEnv.state.mapInfo.max_y\
    or newpos[0] < 0 or newpos[1] < 0:
        return [-1, -1]
    return newpos 
try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    mapID = 4#np.random.randint(1, 6) #Choosing a map ID from 5 maps in Maps folder randomly
    posID_x = 8#np.random.randint(MAP_MAX_X) #Choosing a initial position of the DQN agent on X-axes randomly
    posID_y = 8#np.random.randint(MAP_MAX_Y) #Choosing a initial position of the DQN agent on Y-axes randomly
    #Creating a 3quest for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
    request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
    #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
    minerEnv.send_map_info(request)
    minerEnv.reset()
    s = minerEnv.get_state2(limit)  ##Getting an initial state
    print("Position: ", posID_x, posID_y, "energy: ", minerEnv.state.energy)
    timestep, t, pre_score = 0, 0, 0
    oldpos_gold = [-1, -1]
    history_pos, gold_position = [], []
    for gold in minerEnv.state.mapInfo.golds:
        gold_position.append([gold["posx"], gold["posy"]])
    while not minerEnv.check_terminate():
        try:
            action, value = policy.predict_action(s) # Getting an action from the trained model
            if action == 5 and minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y)<= 0:
                action = 4
                #action = no_craft_gold2()
                print("rest")
            if check_gold() < 4 and [minerEnv.state.x, minerEnv.state.y] == oldpos_gold:
                action = check_gold()
                print("the gold next to, ", action)
            ######   
            t +=1
            if (t > 9 and minerEnv.state.score == pre_score):
                print("craft")
                action = no_craft_gold2()
                #s = minerEnv.get_state2(limit)
                #pre_score = minerEnv.state.score
            ######
            
            if list(newPos([minerEnv.state.x, minerEnv.state.y], action)) == [-1, -1]:
                action = no_craft_gold2()
            newpos = list(newPos([minerEnv.state.x, minerEnv.state.y], action))
            # if newpos in history_pos[-4: -1] and newpos not in gold_position:
            #     action = no_craft_gold2()
            #     print(action)
            if minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) > 0:
                action = 5
                oldpos_gold = [minerEnv.state.x, minerEnv.state.y]
            

            if minerEnv.state.energy  < 9 and action != 4:
                action  = 4
             
            

            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            s_next = minerEnv.get_state2(limit)  # Getting a new state
            reward = minerEnv.get_reward()  # Getting a reward
            print("next action = ", action, "reward = ", reward, "energy = ", minerEnv.state.energy)
            print("score= ", minerEnv.state.score)
            s = s_next
            print(s)
            timestep +=1

            #######
            if action ==5:
                pre_score = minerEnv.state.score
                t = 0
            #######
            
            total_reward += reward
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break
    print(status_map[minerEnv.state.status])
except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")
print("score= ", minerEnv.state.score, "timestep = ", timestep)
for player in minerEnv.state.players:
    print("player", player["playerId"], "---score: ", player["score"], "--energy: ", player["energy"])

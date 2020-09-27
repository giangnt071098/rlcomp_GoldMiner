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
import rule_based

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

try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    mapID = np.random.randint(1, 6) #Choosing a map ID from 5 maps in Maps folder randomly
    posID_x = np.random.randint(MAP_MAX_X) #Choosing a initial position of the DQN agent on X-axes randomly
    posID_y = np.random.randint(MAP_MAX_Y) #Choosing a initial position of the DQN agent on Y-axes randomly
    #Creating a 3quest for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
    request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
    #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
    minerEnv.send_map_info(request)
    minerEnv.reset()
    rulebase = rule_based.RuleBased(minerEnv, [minerEnv.state.x, minerEnv.state.y])
    s = minerEnv.get_state2(limit)  ##Getting an initial state
    print("Position: ", posID_x, posID_y, "energy: ", minerEnv.state.energy)
    timestep, t, pre_score = 0, 0, 0
    oldpos_gold = [-1, -1]
    history_pos = []
    gold_remove = [-1, -1]
    while not minerEnv.check_terminate():
        try:
            # update recent position
            rulebase.update_oldpos([minerEnv.state.x, minerEnv.state.y])
            golds, golds_amount = [], 0
            for gold in minerEnv.state.mapInfo.golds:
                golds.append([gold["posx"], gold["posy"]])
                golds_amount += minerEnv.state.mapInfo.gold_amount(gold["posx"], gold["posy"])
            if gold_remove in golds:
                golds.remove(gold_remove)
            #action, value = policy.predict_action(s) # Getting an action from the trained model
            if timestep <=85:
                golds = rule_based.gold_divide(golds, minerEnv, golds_amount)
            golds_caution = golds.copy()
            action = rulebase.no_craft_gold2(golds)
            print(" RL: ", action)
            if action == 5 and minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y)<= 0:
                #action = 4
                action = rulebase.no_craft_gold2(golds)
                print("rest")
            ######   
            t +=1
            if (t > 6 and minerEnv.state.score == pre_score):
                print("craft")
                action = rulebase.no_craft_gold2(golds)
                #s = minerEnv.get_state2(limit)
                #pre_score = minerEnv.state.score
            ######
            
            # if list(rulebase.newPos(action, rulebase.oldpos)) == [-1, -1]:
            #     action = rulebase.no_craft_gold2()

            # Caution trap
            action = rulebase.caution_trap(action, golds_caution)
            # if rulebase.check_gold() < 4 and timestep >80:
            #     action = rulebase.check_gold()
            #     print("the gold next to, ", action)
            #newpos = list(rulebase.newPos(action, rulebase.oldpos))
            # if newpos in history_pos[-4: -1] and newpos not in gold_position:
            #     action = no_craft_gold2()
            #     print(action)
            gold_target_amount = minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y)
            if gold_target_amount > 0:
                if rulebase.change_gold_position([minerEnv.state.x, minerEnv.state.y], golds_caution, gold_target_amount) == True and timestep <50 and minerEnv.state.lastAction == 5:
                    action = action
                    print("change_action: ", action)
                    gold_remove = [minerEnv.state.x, minerEnv.state.y]
                else: action = 5

            if minerEnv.state.energy  < 6 and action != 4:
                action  = 4
            if minerEnv.state.energy < 18 and minerEnv.state.lastAction == 4 and timestep < 98 and\
            (minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) >150 or minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) ==0):
                action = 4 
            if minerEnv.state.energy < 33 and minerEnv.state.lastAction == 4 and \
            (minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) >500 or minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) ==0):
                action = 4
            # newpos = list(rulebase.newPos(action, rulebase.oldpos))
            # if newpos != [minerEnv.state.x, minerEnv.state.y]:
            #     history_pos.append(list(newpos))
            # if len(history_pos) >10: history_pos.pop(0)
            
            # if timestep == 1: action =1
            # if timestep == 4: action=1
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            print(minerEnv.state.x, minerEnv.state.y, minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y))
            s_next = minerEnv.get_state2(limit)  # Getting a new state
            reward = minerEnv.get_reward()  # Getting a reward
            print("next action = ", action, "reward = ", reward, "energy = ", minerEnv.state.energy)
            print("score= ", minerEnv.state.score)
            s = s_next
            print(s)
            rulebase.update_swamp()
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
print("End game. MapID: ", mapID)
print("score= ", minerEnv.state.score, "timestep = ", timestep)
for player in minerEnv.state.players:
    print("player", player["playerId"], "---score: ", player["score"], "--energy: ", player["energy"])

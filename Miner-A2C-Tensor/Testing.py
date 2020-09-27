from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import torch
from MinerEnv import MinerEnv
import numpy as np
import TD3_conv

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
scale = 3
INPUTNUM = (24, 21*scale, 9*scale)
state_dim = INPUTNUM
action_dim = 6
max_action = 1.0

# load model
kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }
policy = TD3_conv.TD3(**kwargs)
policy_file = "TD3_conv_Miner_0_2_scale3"
policy.load(f"./models_TD3_tensor/{policy_file}")
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
    #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
    request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
    #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
    minerEnv.send_map_info(request)
    minerEnv.reset()
    state = minerEnv.get_state_tensor(scale)  ##Getting an initial state
    observ = np.reshape([np.stack((state, state, state, state), axis = 0)], (24, INPUTNUM[1], INPUTNUM[2]))
    print("Position: ", posID_x, posID_y, "energy: ", minerEnv.state.energy)
    timestep = 0
    while not minerEnv.check_terminate():
        try:
            action, _ = policy.predict_action(observ) # Getting an action from the trained model
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            next_state = minerEnv.get_state_tensor(scale)  # Getting a new state
            next_observ = np.append(np.reshape(next_state, (6, INPUTNUM[1], INPUTNUM[2])), observ[:18, :, :], axis = 0)
            reward = minerEnv.get_reward()  # Getting a reward
            print("next action = ", action, "reward = ", reward, "energy = ", minerEnv.state.energy)
            print("score= ", minerEnv.state.score)
            observ = next_observ
            #print(next_state)
            timestep +=1
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

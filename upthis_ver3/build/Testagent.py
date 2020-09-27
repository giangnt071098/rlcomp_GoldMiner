from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import torch
import numpy as np
import TD3_bot
from MinerEnv import MinerEnv
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
policy.load(f"./models/{policy_file}")
print("Loaded model from disk")
status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
                  3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"} 
try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    minerEnv.reset()
    rulebase = rule_based.RuleBased(minerEnv, [minerEnv.state.x, minerEnv.state.y])
    s = minerEnv.get_state2(limit)  ##Getting an initial state
    t, pre_score = 0, 0
    oldpos_gold = [-1, -1]
    while not minerEnv.check_terminate():
        try:
            # update recent position
            rulebase.update_oldpos([minerEnv.state.x, minerEnv.state.y])
            golds = []
            for gold in minerEnv.state.mapInfo.golds:
                golds.append([gold["posx"], gold["posy"]])
            t += 1
            golds = rule_based.gold_divide(golds)
            action = rulebase.no_craft_gold2(golds)
            #action, _ = policy.predict_action(s) # Getting an action from the trained model
            if action == 5 and minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) <=0:
                #action = 4
                action = rulebase.no_craft_gold2(golds)
                print("rest")
            ######   
            if (t > 6 and minerEnv.state.score == pre_score):
                print("craft")
                action = rulebase.no_craft_gold2(golds)
            ######

            # Caution trap
            action = rulebase.caution_trap(action, golds)
            if rulebase.check_gold() < 4:
                action = rulebase.check_gold()
                print("the gold next to, ", action)
            if minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) > 0 and action != 5:
                action = 5
                oldpos_gold = [minerEnv.state.x, minerEnv.state.y]
            if minerEnv.state.energy  < 6 and action != 4:
                action  = 4
            if minerEnv.state.energy < 18 and minerEnv.state.lastAction == 4 and \
            (minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) >150 or minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) ==0):
                action = 4
            if minerEnv.state.energy < 31 and minerEnv.state.lastAction == 4 and \
            (minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) >500 or minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) ==0):
                action = 4
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            s_next = minerEnv.get_state2(limit)  # Getting a new state
            reward = minerEnv.get_reward()  # Getting a reward
            print("next action = ", action, "energy = ", minerEnv.state.energy, "score: ", minerEnv.state.score)
            s = s_next
            rulebase.update_swamp()
            #######
            if action ==5:
                pre_score = minerEnv.state.score
                t = 0
            #######
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break
    print(status_map[minerEnv.state.status])
except Exception as e:
    import traceback
    traceback.print_exc()
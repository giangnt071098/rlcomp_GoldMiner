from MINER_STATE import State
import numpy as np
from models_TD3_2 import TD3_bot

class PlayerInfo:
    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = 0
        self.freeCount = 0


class Bot_TD3_2:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5


    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)
        self.limit = 2
        state_dim = (2*self.limit+1)**2 + 3
        action_dim = 6
        max_action = 1.0
        # load model
        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                }
        policy_file = "TD3_Miner_0_2"
        self.TreeID = 1
        self.TrapID = 2
        self.SwampID = 3
        self.policy = TD3_bot.TD3(**kwargs)
        self.policy.load(f"./models_TD3_2/{policy_file}")
        
    def next_action(self):
        s = self.get_state2(self.limit)
        action, _ = self.policy.predict_action(s)
        return int(action)
    def get_state2(self, limit):
        # Building the map
        view = np.zeros([limit*2+1, limit*2+1], dtype=int)
        max_x, max_y = self.state.mapInfo.max_x, self.state.mapInfo.max_y
        xlimit_below = np.clip(self.info.posx - limit, 0, max_x) - np.clip(self.info.posx + limit - max_x, 0, limit)
        xlimit_up = np.clip(self.info.posx + limit, 0, max_x) + np.clip(0 - self.info.posx + limit, 0, limit)
        ylimit_below = np.clip(self.info.posy - limit, 0, max_y) - np.clip(self.info.posy + limit - max_y, 0, limit)
        ylimit_up = np.clip(self.info.posy + limit, 0, max_y) + np.clip(0 - self.info.posy + limit, 0, limit)

        #print(xlimit_below, xlimit_up, ylimit_below, ylimit_up, self.info.posx, self.info.posy)

        for i in range(xlimit_below, xlimit_up + 1):
            for j in range(ylimit_below, ylimit_up + 1):
                if self.state.mapInfo.get_obstacle(i, j) == self.TreeID:  # Tree
                    view[i - xlimit_below, j - ylimit_below] = -self.TreeID
                if self.state.mapInfo.get_obstacle(i, j) == self.TrapID:  # Trap
                    view[i - xlimit_below, j - ylimit_below] = -self.TrapID
                if self.state.mapInfo.get_obstacle(i, j) == self.SwampID: # Swamp
                    view[i - xlimit_below, j - ylimit_below] = -self.SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i - xlimit_below, j - ylimit_below] = self.state.mapInfo.gold_amount(i, j)/10
        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector

        # Add position and energy of agent to the DQNState
        DQNState.append(self.info.posx - xlimit_below)
        DQNState.append(self.info.posy - ylimit_below)
        DQNState.append(self.info.energy)
        #Add position of bots 
        # for player in self.state.players:
        #     if player["playerId"] != self.state.id:
        #         DQNState.append(player["posx"])
        #         DQNState.append(player["posy"])
                
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)
        return DQNState
    def new_game(self, data):
        try:
            self.state.init_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def new_state(self, data):
        # action = self.next_action();
        # self.socket.send(action)
        try:
            self.state.update_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
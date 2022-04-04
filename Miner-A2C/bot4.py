from MINER_STATE import State
import numpy as np


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


class Bot4:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)
    def distance(self, posA, posB):
        return (posA[0] - posB[0])**2 + (posA[1] - posB[1])**2

    def next_action(self):
        golds = self.state.mapInfo.golds
        mindis = 1000
        bot_posx, bot_posy = self.info.posx, self.info.posy
        for gold in golds:
            dist =self.distance([gold["posx"], gold["posy"]], [bot_posx, bot_posy])
            if  dist < mindis:
                mindis = dist
                target_x = gold["posx"]
                target_y = gold["posy"]
        if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:
            if self.info.energy >= 6:
                return self.ACTION_CRAFT
        if self.info.energy <10:
                return self.ACTION_FREE
        if (target_x - bot_posx) < 0:
            return self.ACTION_GO_LEFT
        if (target_x - bot_posx) > 0:
            return self.ACTION_GO_RIGHT
        if (target_y - bot_posy) < 0:
            return self.ACTION_GO_UP
        if (target_y - bot_posy) > 0:
            return self.ACTION_GO_DOWN
        return self.np.random.randrange(0, 4)

        # if self.info.posx == 9:
        #     return 4
        # return 0

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
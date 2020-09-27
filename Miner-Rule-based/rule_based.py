import numpy as np
TreeID = 1
TrapID = 2
SwampID = 3
max_x = 20
max_y = 8 
def distance(posA, posB):
    return abs(posA[0] - posB[0]) + abs(posA[1] - posB[1])
def gold_divide(golds, minerEnv, golds_amount):
	## when half of map has only a few gold position -> remove all in it
	gold_left = []
	gold_amount_left = 0
	gold_right = []
	gold_amount_right = 0
	for gold in golds:
		if gold[0] < max_x//2-4:
			gold_left.append(gold)
			gold_amount_left += minerEnv.state.mapInfo.gold_amount(gold[0], gold[1])
		elif gold[0] > max_x //2+4:
			gold_right.append(gold)
			gold_amount_right += minerEnv.state.mapInfo.gold_amount(gold[0], gold[1])
	if golds_amount - 5*gold_amount_left >0: 
	#if len(golds) - 6*len(gold_left) >=0:
		golds = [x for x in golds if (x not in gold_left)]
		return golds
	if golds_amount - 5*gold_amount_right >0:
	#if len(golds)- 6*len(gold_right) >=0:
		golds = [x for x in golds if (x not in gold_right)]
		return golds
	return golds

class RuleBased(object):
	def __init__(self, minerEnv, oldpos):
		self.minerEnv = minerEnv
		self.oldpos = oldpos
		self.Swamp_position = {1:[], 2:[], 3:[]}
		self.energy = 50
		self.last2pos = []
		self.lastpos = oldpos
		self.lastgold = oldpos
		self.players = {1:[], 2:[], 3:[]}
		self.lastplayer_pos = {0:[-1,-1],1:[-1,-1],2:[-1,-1],3:[-1,-1]}
	def no_craft_gold2(self, golds):
		#them tu day
		action_x, action_y, reward_x, reward_y = -1, -1, 0.0, 0.0
		direct_x, direc_y = 0, 0
		#golds = self.minerEnv.state.mapInfo.golds
		mindis = 1000
		maxdis =-1000
		miner_posx, miner_posy = self.oldpos[0], self.oldpos[1]
		target_x, target_y = miner_posx, miner_posy
		gold_target = 0
		for gold in golds:
			dist = distance([gold[0], gold[1]], [miner_posx, miner_posy]) - self.reward_gold([gold[0], gold[1]])
			if  dist < mindis:
				mindis = dist
				target_x = gold[0]
				target_y = gold[1]
				gold_target = self.minerEnv.state.mapInfo.gold_amount(target_x, target_y)
			if dist == mindis and self.minerEnv.state.mapInfo.gold_amount(gold[0], gold[1]) > gold_target:
				mindis = dist
				target_x = gold[0]
				target_y = gold[1]
				gold_target = self.minerEnv.state.mapInfo.gold_amount(target_x, target_y)
		if (target_x - miner_posx) < 0:
			action_x, direct_x = 0, -1
		if (target_x - miner_posx) > 0:
			action_x, direct_x = 1, 1
		if (target_y - miner_posy) < 0:
			action_y, direc_y = 2, -1
		if (target_y - miner_posy) > 0:
			action_y, direc_y = 3, 1
		# if other players in same gold position -> go to other gold
		if self.change_gold_position([target_x, target_y], golds, gold_target):
			golds.remove([target_x, target_y])
			return self.no_craft_gold2(golds)
		####
		if action_x == -1 and action_y != -1: 
			if list(self.newPos(action_y, self.oldpos)) in (self.Swamp_position[1] + self.Swamp_position[2] +self.Swamp_position[3]):
				print("changey")
				golds.remove([target_x, target_y])
				return self.no_craft_gold2(golds)
			else: return action_y
		if action_y == -1 and action_x != -1: 
			if list(self.newPos(action_x, self.oldpos)) in (self.Swamp_position[1] + self.Swamp_position[2] +self.Swamp_position[3]):
				print("changex")
				golds.remove([target_x, target_y])
				return self.no_craft_gold2(golds)
			else: return action_x
		if action_x != -1 and action_y != -1:
			if list(self.newPos(action_x, self.oldpos)) in (self.Swamp_position[1] + self.Swamp_position[2] +self.Swamp_position[3])\
			or list(self.newPos(action_y, self.oldpos)) in (self.Swamp_position[1] + self.Swamp_position[2] +self.Swamp_position[3]):
				print("changexy")
				golds.remove([target_x, target_y])
				return self.no_craft_gold2(golds)
				
			range_ = min(abs(target_x - miner_posx), abs(target_y - miner_posy))
			for i in range(range_):
				reward_x += self.rule_reward(self.newPos(action_x, [miner_posx + i*direct_x, miner_posy]))*(0.9**i)
				print("x", reward_x, miner_posx + (i+1)*direct_x, miner_posy)
			#reward_x = self.rule_reward(self.newPos(action_x, [miner_posx, miner_posy]))
			for j in range(range_):
				reward_y += self.rule_reward(self.newPos(action_y, [miner_posx, miner_posy + j*direc_y]))*(0.9**j)
				print("y", reward_y, miner_posx, miner_posy + (j+1)*direc_y)
			#reward_y = self.rule_reward(self.newPos(action_y, [miner_posx, miner_posy]))
			print(reward_x, reward_y)
			if reward_x >= reward_y:
				print("x")
				return action_x
			print("y")
			return action_y
		return 4
	def go_to_gold(self, golds):
		#them tu day
		action_x, action_y, reward_x, reward_y = -1, -1, 0.0, 0.0
		direct_x, direc_y = 0, 0
		#golds = self.minerEnv.state.mapInfo.golds
		mindis = 1000
		miner_posx, miner_posy = self.oldpos[0], self.oldpos[1]
		target_x, target_y = miner_posx, miner_posy
		for gold in golds:
			dist = distance([gold[0], gold[1]], [miner_posx, miner_posy]) - self.reward_gold([gold[0], gold[1]])
			if  dist < mindis:
				mindis = dist
				target_x = gold[0]
				target_y = gold[1]
		if (target_x - miner_posx) < 0:
			action_x, direct_x = 0, -1
		if (target_x - miner_posx) > 0:
			action_x, direct_x = 1, 1
		if (target_y - miner_posy) < 0:
			action_y, direc_y = 2, -1
		if (target_y - miner_posy) > 0:
			action_y, direc_y = 3, 1
		if action_x == -1 and action_y != -1:  return action_y
		if action_y == -1 and action_x != -1:  return action_x
		if action_x != -1 and action_y != -1:
			range_ = min(abs(target_x - miner_posx), abs(target_y - miner_posy))
			for i in range(range_):
				reward_x += self.rule_reward(self.newPos(action_x, [miner_posx + i*direct_x, miner_posy]))*(0.9**i)
				print("x", reward_x, miner_posx + (i+1)*direct_x, miner_posy)
			#reward_x = self.rule_reward(self.newPos(action_x, [miner_posx, miner_posy]))
			for j in range(range_):
				reward_y += self.rule_reward(self.newPos(action_y, [miner_posx, miner_posy + j*direc_y]))*(0.9**j)
				print("y", reward_y, miner_posx, miner_posy + (j+1)*direc_y)
			#reward_y = self.rule_reward(self.newPos(action_y, [miner_posx, miner_posy]))
			print(reward_x, reward_y)
			if reward_x >= reward_y:
				return action_x
			return action_y
		return 4
	def check_gold(self):
	    x,y = self.oldpos[0], self.oldpos[1]
	    max_gold, pre_action = 0, 4
	    for stt, (i, j) in enumerate(zip([-1,1,0,0],[0,0,-1,1])):
	        xnew, ynew = x+i, y+j
	        if xnew <= self.minerEnv.state.mapInfo.max_x and xnew >=0 \
	        and ynew <= self.minerEnv.state.mapInfo.max_y and ynew >= 0:
	            if self.minerEnv.state.mapInfo.gold_amount(xnew, ynew) > max_gold:
	                pre_action, max_gold = stt, self.minerEnv.state.mapInfo.gold_amount(xnew, ynew)
	    return pre_action if pre_action != 4 else 10
	def newPos(self, action, oldpos):
	    action_dict = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1], 4: [0, 0], 5: [0, 0]}
	    newpos = np.add(oldpos, action_dict[action])
	    if newpos[0] > self.minerEnv.state.mapInfo.max_x or newpos[1] > self.minerEnv.state.mapInfo.max_y\
	    or newpos[0] < 0 or newpos[1] < 0:
	        return [-1, -1]
	    return newpos
	def rule_reward(self, pos):
		# Calculate reward
		reward = 0
		#If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
		if self.minerEnv.state.mapInfo.get_obstacle(pos[0], pos[1]) == TreeID:  # Tree
			reward -= 0.25
		if self.minerEnv.state.mapInfo.get_obstacle(pos[0], pos[1]) == TrapID:  # Trap
			reward -= 0.2
		if self.minerEnv.state.mapInfo.get_obstacle(pos[0], pos[1]) == SwampID:  # Swamp
			if [pos[0], pos[1]] in self.Swamp_position[1]:           
				reward -= 0.5
			if [pos[0], pos[1]] in self.Swamp_position[2]:
				reward -= 1
			if [pos[0], pos[1]] in self.Swamp_position[3]:
				reward -= 2
			else:
				reward -= 0.1 
		if len(self.last2pos)== 2 and self.minerEnv.state.mapInfo.gold_amount(pos[0], pos[1]) <=0:
			if [pos[0], pos[1]] == self.lastpos and self.last2pos[1] != self.lastgold:
				reward -= 0.8
		return reward
	def reward_gold(self, gold_pos):
	    x,y = gold_pos[0], gold_pos[1]
	    reward = 0
	    for stt, (i, j) in enumerate(zip([-1,1,0,0,-1,1,-1,1],[0,0,-1,1,-1,-1,1,1])):
	        xnew, ynew = x+i, y+j
	        if xnew <= self.minerEnv.state.mapInfo.max_x and xnew >=0 \
	        and ynew <= self.minerEnv.state.mapInfo.max_y and ynew >= 0:
	        	amount = self.minerEnv.state.mapInfo.gold_amount(xnew, ynew)
	        	if  amount>=100 and amount<= 200:
	        		reward += 1
	        	if amount >200 and amount <=500:
	        		reward += 2
	        	if amount >500:
	        		reward += 3
	        	if amount >=1000:
	           		reward += 5
	    return reward
	def change_gold_position(self, gold_position, golds, gold_target):
		if gold_target <=100 and [gold_position[0], gold_position[1]] in self.players[1] and len(golds)>1:
			return True
		if gold_target <=200 and [gold_position[0], gold_position[1]] in self.players[2] and len(golds)>1:
			return True
		if gold_target <=300 and [gold_position[0], gold_position[1]] in self.players[3] and len(golds)>1:
			return True
		return False
	def caution_trap(self, action, golds):
		if self.energy > 40 and action == 4:
			action = self.go_to_gold(golds)
		if list(self.newPos(action, self.oldpos)) == [-1, -1]:
			action = self.no_craft_gold2(golds)
		# if newpos is Swamp or back to oldpos
		if len(self.last2pos) ==2:
			if list(self.newPos(action, self.oldpos)) == self.last2pos[0] and list(self.newPos(action, self.oldpos)) != self.last2pos[1]:
				action = self.no_craft_gold2(golds)

		if list(self.newPos(action, self.oldpos)) in (self.Swamp_position[1] + self.Swamp_position[2]):
			if self.energy <=40: action = 4
		if list(self.newPos(action, self.oldpos)) in self.Swamp_position[3]:
			action = 4
		if self.minerEnv.state.mapInfo.get_obstacle(*self.newPos(action, self.oldpos)) == TreeID and self.energy <=20:
			action = 4
		if self.minerEnv.state.mapInfo.get_obstacle(*self.newPos(action, self.oldpos)) == TrapID and self.energy <=10:
			action = 4

		return action

	def update_oldpos(self, oldpos):
		self.oldpos = oldpos
		self.energy = self.minerEnv.state.energy
		
		self.last2pos.append([oldpos[0], oldpos[1]])
		if len(self.last2pos) > 2: self.last2pos.pop(0)
		if self.last2pos[0] != self.oldpos:
			self.lastpos = self.last2pos[0]
		if self.minerEnv.state.lastAction == 5 and len(self.last2pos) == 2:
			self.lastgold = self.last2pos[0]
		
		# if self.minerEnv.state.mapInfo.get_obstacle(oldpos[0], oldpos[1]) == 3:
		# 	self.Swamp_position.append([oldpos[0], oldpos[1]])
		# UPDATE Swamp_position
	def update_swamp(self):
		# UPDATE Swamp_position
		self.players = {1:[], 2:[], 3:[]}
		players_pos = []
		for stt, player in enumerate(self.minerEnv.state.players):
			if self.minerEnv.state.mapInfo.get_obstacle(player["posx"], player["posy"]) == 3 \
			and [player["posx"], player["posy"]] != self.lastplayer_pos[stt]:
				if [player["posx"], player["posy"]] not in self.Swamp_position[2] \
				and [player["posx"], player["posy"]] not in self.Swamp_position[3]\
				and [player["posx"], player["posy"]] not in self.Swamp_position[1]\
				and [player["posx"], player["posy"]] not in players_pos :
					self.Swamp_position[1].append([player["posx"], player["posy"]])
				elif [player["posx"], player["posy"]] in self.Swamp_position[1]\
				and [player["posx"], player["posy"]] not in players_pos:
					try:
						if player["energy"]>0:
							self.Swamp_position[2].append([player["posx"], player["posy"]])
							self.Swamp_position[1].remove([player["posx"], player["posy"]])
					except:
						pass
				elif [player["posx"], player["posy"]] in self.Swamp_position[2]\
				and [player["posx"], player["posy"]] not in players_pos:
					try:
						if player["energy"]>0:
							self.Swamp_position[3].append([player["posx"], player["posy"]])
							self.Swamp_position[2].remove([player["posx"], player["posy"]])
					except:
						pass
				players_pos.append([player["posx"], player["posy"]])
			self.lastplayer_pos[stt] = [player["posx"], player["posy"]]
			if player["playerId"] != self.minerEnv.state.id:
				if [player["posx"], player["posy"]] not in self.players[2] \
				and [player["posx"], player["posy"]] not in self.players[3]\
				and [player["posx"], player["posy"]] not in self.players[1]:
					self.players[1].append([player["posx"], player["posy"]])
				elif [player["posx"], player["posy"]] in self.players[1]:
					try:
						if player["energy"]>0:
							self.players[2].append([player["posx"], player["posy"]])
							self.players[1].remove([player["posx"], player["posy"]])
					except:
						pass
				elif [player["posx"], player["posy"]] in self.players[2]:
					try:
						if player["energy"]>0:
							self.players[3].append([player["posx"], player["posy"]])
							self.players[2].remove([player["posx"], player["posy"]])
					except:
						pass
		print(self.Swamp_position, self.players, self.energy)

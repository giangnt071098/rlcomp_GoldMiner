import sys
from DQNModel import DQN # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import Memory # A class of creating a batch in order to store experiences for the training process
from bot1 import Bot1
from bot2 import Bot2
from bot3 import Bot3

import pandas as pd
import datetime 
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", default="")
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", default="1111")
parser.add_argument("--notice_name", default= "")
args = parser.parse_args()
HOST = str(args.host)
PORT = str(args.port)
limit=2

# HOST = "localhost"
# PORT = 1111
# if len(sys.argv) == 3:
#     HOST = str(sys.argv[1])
#     PORT = int(sys.argv[2])

# Create header for saving DQN learning file
now = datetime.datetime.now() #Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon", "Done", "Termination_Code"] #Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + args.notice_name + ".csv" 
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

# Parameters for training a DQN model
N_EPISODE = 100000 #The number of episodes for training
MAX_STEP = 1000   #The number of steps for each episode
BATCH_SIZE = 32   #The number of experiences for each replay 
MEMORY_SIZE = 100000 #The size of the batch for storing experiences
SAVE_NETWORK = 1000  # After this number of episodes, the DQN model is saved for testing later. 
INITIAL_REPLAY_SIZE = 10000 #The number of experiences are stored in the memory batch before starting replaying
INPUTNUM = (2*limit+1)**2 + 3#198 #The number of input values for the DQN model
ACTIONNUM = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map

# Initialize a DQN model and a memory batch for storing experiences
DQNAgent = DQN(INPUTNUM, ACTIONNUM)
memory = Memory(MEMORY_SIZE)
bots = [Bot1(2), Bot2(3), Bot3(4)]
#load model to continue training
if args.load_model !="":
    file_name = "TrainedModels/DQNmodel_20200730-1832_ep1000out-30.json"
    json_file = file_name if args.load_model == "default" else args.load_model
    DQNAgent.load_model(json_file)

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game

train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.
#Training Process
#the main part of the deep-q learning agorithm 
for episode_i in range(0, N_EPISODE):
    print(np.unique(DQNAgent.model.get_weights()[0]))
    try:
        # Choosing a map in the list
        mapID = np.random.randint(1, 6) #Choosing a map ID from 5 maps in Maps folder randomly
        posID_x = np.random.randint(MAP_MAX_X) #Choosing a initial position of the DQN agent on X-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y) #Choosing a initial position of the DQN agent on Y-axes randomly
        #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
        #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)

        # Getting the initial state
        minerEnv.reset() #Initialize the game environment
        s = minerEnv.get_state2(limit)#Get the state after reseting. 
                                #This function (get_state()) is an example of creating a state for the DQN model 
        total_reward = 0 #The amount of rewards for the entire episode
        terminate = False #The variable indicates that the episode ends
        maxStep = minerEnv.state.mapInfo.maxStep #Get the maximum number of steps for each episode in training
        average_loss = 0
        #Here: maxStep =100 line 64 GAME_SOCKET_DUMMY
        #Start an episde for training
        for step in range(0, maxStep):
            #s.shape =(MAP_MAX_X, MAP_MAX_Y) moi phan tu la gia tri score VD 0, -1, 100, 250, -3
            if args.load_model != "":
                action = np.argmax(DQNAgent.model.predict(s.reshape(1,len(s))))
            else:
                # _id = np.random.randint(1,5)
                # if _id == 1:
                #     action = DQNAgent.act(s)  # Getting an action from the DQN model from the state (s)
                # else: action = bots[_id-2].act_sample(s)
                action = DQNAgent.act(s)
            #action int, VD action =3
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            s_next = minerEnv.get_state2(limit)  # Getting a new state
            #s_next tuong tu s
            reward = minerEnv.get_reward()  # Getting a reward
            #reward int, VD 0
            terminate = minerEnv.check_terminate()  # Checking the end status of the episode
            #terminal True or False
            # Add this transition to the memory batch
            memory.push(s, action, reward, terminate, s_next)

            # Sample batch memory to train network
            if (memory.length > INITIAL_REPLAY_SIZE):
                #If there are INITIAL_REPLAY_SIZE experiences in the memory batch
                #then start replaying
                batch = memory.sample(BATCH_SIZE) #Get a BATCH_SIZE experiences for replaying
                #batch list 5, batch[0] (32, 198), list[1] (32,) batch[2] (32,), batch[3] (32, 198), batch[4] (32,)
                DQNAgent.replay(batch, BATCH_SIZE)#Do relaying
                average_loss += DQNAgent.loss_show
                train = True #Indicate the training starts
            total_reward = total_reward + reward #Plus the reward to the total rewad of the episode
            s = s_next #Assign the next state for the next step.

            # Saving data to file
            save_data = np.hstack(
                [episode_i + 1, step + 1, reward, total_reward, action, DQNAgent.epsilon, terminate]).reshape(1, 7)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)
            
            if terminate == True:
                #If the episode ends, then go to the next episode
                break

        # Iteration to save the network architecture and weights
        if (np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True):
            DQNAgent.target_train()  # Replace the learning weights for target model with soft replacement
            #Save the DQN model
            now = datetime.datetime.now() #Get the latest datetime
            DQNAgent.save_model("TrainedModels/",
                                "DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1)+ args.notice_name)

        
        #Print the training information after the episode
        print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.2f. Epsilon = %.2f .Termination code: %d' % (
            episode_i + 1, step + 1, total_reward, DQNAgent.epsilon, terminate))
        print('Average loss Episode: %.2f'%(average_loss/(step+1)))
        for player in minerEnv.state.players:
            print("player", player["playerId"], "---score: ", player["score"], "--energy: ", player["energy"])
        # Save information
        f = open("data.txt", "w")
        f.write(str([total_reward, episode_i+1]))
        f.close()
        #Decreasing the epsilon if the replay starts
        if train == True:   
            DQNAgent.update_epsilon()

    except Exception as e:
        import traceback

        traceback.print_exc()
        # print("Finished.")
        break
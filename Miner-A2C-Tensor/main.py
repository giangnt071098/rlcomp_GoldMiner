import numpy as np
import torch
import argparse, random
import os
from MinerEnv import MinerEnv
import utils
import TD3, DDPG, newDDPG, A2C, newTD3

def request_to_env(minerEnv, _train= False):
	if _train == False:
		mapID = np.random.choice(range(1,7), 1, p=[0.15,0.15,0.15, 0.15, 0.15, 0.25])[0] #Choosing a map ID from 5 maps in Maps folder randomly
	else:
		mapID = np.random.randint(1,6)
	posID_x = np.random.randint(MAP_MAX_X) #Choosing a initial position of the DQN agent on X-axes randomly
	posID_y = np.random.randint(MAP_MAX_Y) #Choosing a initial position of the DQN agent on Y-axes randomly
	#Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
	request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
	#Send the request to the game environment (GAME_SOCKET_DUMMY.py)
	minerEnv.send_map_info(request)
	return mapID
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, eval_episodes=10):
	avg_reward = 0.
	for _ in range(eval_episodes):
		_= request_to_env(eval_env, True)
		eval_env.reset()
		state, done = eval_env.get_state2(int(args.limit_obs)), False
		while not done:
			action, _ = policy.select_action(np.array(state))
			eval_env.step(str(action)) 
			next_state = eval_env.get_state2(int(args.limit_obs))
			reward = eval_env.get_reward()
			done = eval_env.check_terminate()
			avg_reward += reward
	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                 # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Miner")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=10e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--batch_size", default=512, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--learning_rate", default=3e-4)
    parser.add_argument("--tau", default=0.005)                      # Target network update rate
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--limit_obs", default=2, type=int)
    parser.add_argument("--n_episode", default=100000, type=int)
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="1111")
    args = parser.parse_args()
    
    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.limit_obs}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    INPUTNUM = (2*(args.limit_obs)+1)**2 +3 +6 #The number of input values for the DQN model
    ACTIONNUM = 6  #The number of actions output from the DQN model
    MAP_MAX_X = 21 #Width of the Map
    MAP_MAX_Y = 9
    SAVE_NETWORK = 2000
    N_EPISODE = int(args.n_episode)
    HOST = str(args.host)
    PORT = str(args.port)
    if not os.path.exists("./results"):
    	os.makedirs("./results")
    if args.save_model and not os.path.exists("./models"):
    	os.makedirs("./models")

	# Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
	
    state_dim = INPUTNUM
    action_dim = ACTIONNUM
    max_action = 1.0
    if args.load_model != "":
        epsilon = 0.01
    else:
        epsilon = 1
    kwargs = {
    	"state_dim": state_dim,
    	"action_dim": action_dim,
    	"max_action": max_action,
    	"discount": args.discount,
    	"tau": float(args.tau),
    	"learning_rate": float(args.learning_rate),
        "epsilon": float(epsilon)
    }

    # Initialize policy

    if args.policy == "TD3":
    	# Target policy smoothing is scaled wrt the action scale
    	kwargs["policy_freq"] = args.policy_freq
    	policy = TD3.TD3(**kwargs)
    if args.policy == "DDPG":
    	policy = DDPG.DDPG(**kwargs)
    if args.policy == "newDDPG":
    	policy = newDDPG.DDPG(**kwargs)
    if args.policy == "newTD3":
    	policy = newTD3.TD3(**kwargs)
    if args.policy == "A2C":
    	policy = A2C.A2C(**kwargs)

    if args.load_model != "":
    	policy_file = file_name if args.load_model == "default" else args.load_model
    	policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim = action_dim)

    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()
    #init environment

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, minerEnv)]
    train = False
    for episode_i in range(0, N_EPISODE):
        # Reset environment
        mapID = request_to_env(minerEnv, train)
        # init environment game
        minerEnv.reset()
        #action = policy.select_action(np.array(state))
        state = minerEnv.get_state2(int(args.limit_obs))
        done = False
        maxStep = minerEnv.state.mapInfo.maxStep
        total_reward = 0
    	#print(action, reward, done_bool)
    	# Train agent after collecting sufficient data
        for step in range(0, maxStep):
            action, action_probs = policy.select_action(np.array(state))
            minerEnv.step(str(action))
            next_state = minerEnv.get_state2(int(args.limit_obs))
            reward = minerEnv.get_reward()
            done = minerEnv.check_terminate()
            done_bool = float(done)
            replay_buffer.add(state, action_probs, next_state, reward, done_bool)
            if replay_buffer.size >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
                train = True
            total_reward = total_reward + reward
            state = next_state
            if done:
                break
                
        if (np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True):
            if args.save_model: policy.save(f"./models/{file_name}")
        # Evaluate episode
        if episode_i % 100 == 0:
            evaluations.append(eval_policy(policy, minerEnv))
            np.save(f"./results/{file_name}", evaluations)
                

        #Print the training information after the episode
        print(np.unique(policy.critic.l1.weight.cpu().data))
        print(policy.predict_action(state)[1])
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Episode Num: {episode_i+1} Episode T: {step+1} Reward: {total_reward:.3f} epsilon: {policy.epsilon: .3f} terminate: {done}")
        print("Actor loss = ", policy.loss_actor, "Critic loss = ", policy.loss_critic)
        for player in minerEnv.state.players:
            print("player", player["playerId"], "---score: ", player["score"], "--energy: ", player["energy"])

        if train == True:
            policy.update_epsilon()


			
		

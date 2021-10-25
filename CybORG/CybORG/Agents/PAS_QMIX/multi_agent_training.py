from tqdm import tqdm
from CybORG.Agents.PAS_QMIX.agents import Dueling_Agent
from CybORG.Agents.PAS_QMIX.networks import DuelingQNetwork, ParamNetwork


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 


from torch.utils.tensorboard import SummaryWriter


import numpy as np 
import gym 
import random
from tqdm import tqdm

from collections import namedtuple, deque



from CybORG import CybORG
import inspect

from CybORG.Agents import TestAgent
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper

from arguments import parse_args




def one_hot_encoding(q_values):
    output = torch.zeros_like(q_values)
    max_index = torch.argmax(q_values).item()
    output[max_index] = 1

    return output


def detect_host(host_list, current_host):
    if len(host_list) != len(current_host):
        for h in host_list:
            if h not in current_host:
                current_host.add(h)

        return current_host, True
    
    return current_host, False


def numpy_to_torch(np_arr):
    assert isinstance(np_arr, np.ndarray), "input must be a numpy array"
    if isinstance(np_arr, torch.Tensor):
        return np_arr
    return torch.tensor(np_arr, dtype=torch.float64) 


def argmax_rand(arr):
	# np.argmax with random tie breaking
	return np.random.choice(np.flatnonzero(np.isclose(arr, np.max(arr), atol=1e-3)))


def train(agent_list, original_selection_masks, valid_action_index, full_action_space, n_episodes, max_step, expected_flag = 1, eps_start=1.0, eps_end=0.1, eps_decay=0.99):

    #stats = utils.Stats(num_episodes=n_episodes, continuous=True)
    scores = []
    steps = []
    scores_window = deque(maxlen=100)
    
    eps = eps_start

    for i_episode in range(n_episodes):
        print("\rEpisode: {}".format(i_episode))
        score = 0
        flags_collected = 0
        agent_name = 'Red'
        step = 0
        selection_masks = original_selection_masks
        observation = cyborg.reset(agent=agent_name)
        current_host = set(selection_masks[7]) # compromised host id

        for t in range(max_step):
            
            observation = numpy_to_torch(observation)
            
            
            actions = np.array([0 for _ in range(len(full_action_space))])
            
            action_mask = [selection_masks[i] for i in valid_action_index]

            rl_actions = []
            for i, mask in enumerate(action_mask):
                act = agent_list[i].act(observation, mask, eps)
                rl_actions.append(act)

            actions[valid_action_index] = rl_actions
            
            next_observation, reward, done, info = cyborg.step(action=list(actions))
            selection_masks = info["selection_masks"]
            
            current_host, new_host = detect_host(selection_masks[7], current_host)

            if reward == 0:
                reward -= 0.0001

            if new_host:
                #import pdb; pdb.set_trace()
                #print("new host")
                reward += 0.01
            
            

            if reward == 10:
                flags_collected += 1
                score += reward
                if flags_collected == expected_flag:
                    done = True
                    flags_collected = 0
                    #rl_agent.step(observation, actions[valid_action_index], reward, next_observation, done)
                    for i, agent in enumerate(agent_list):
                        agent.step(observation, rl_actions[i], reward, next_observation, done)
                    print("Capture flat at:", t)
                    
                    break
            
            step = t
            
            for i, agent in enumerate(agent_list):
                agent.step(observation, rl_actions[i], reward, next_observation, done)
            
            observation = next_observation
            score += reward
        
        
        if not done:
            print("Failed at step:", step)    
        scores_window.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\t Eps {}\tAverage Score: {:.2f}'.format(i_episode, eps, np.mean(scores_window)), end="")


if __name__ == '__main__':
    # Load environment
    import time
    seed = 42
    experiment_name = f"cyborg_marl_{seed}__{int(time.time())}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    writer = SummaryWriter(f"runs/{experiment_name}")

    print(torch.cuda.get_device_name())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    args = parse_args()

    args.device = device

    print("device:", args.device)
    scenario_name = "Scenario1"
    scenario_name = "scenario_12_hosts_2flag"
    
    print(scenario_name)
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario_name}.yaml'

    agent_name = 'Red'

    
    cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))

    observation = cyborg.env.reset(agent=agent_name)
    full_action_space = cyborg.get_action_space(agent_name) # [6, 5, 13, 6, 69, 4, 8, 13]
    
    valid_action_index = np.array([0,1,2,7])
    valid_action_space = np.array(full_action_space)[valid_action_index] # [6,5,13,13]
    
    #original_selection_masks = [observation.selection_masks[i] for i in [0,1,2,-1]]
    original_selection_masks = observation.selection_masks
    print(original_selection_masks)
    print(full_action_space)
    print(valid_action_space)
    print(len(observation.observation))

    
    obs_torch = torch.tensor(observation.observation, dtype=torch.float32)
    print(obs_torch.shape)
    
    # Load agents
    # 1. Create action agent: normal dueling dqn agent
    #import pdb; pdb.set_trace()
    agent_list = [Dueling_Agent(obs_torch.shape[0], valid_action, args) for valid_action in valid_action_space ]
    
    for i, agent in enumerate(agent_list):
        print(f"Agent {i+1} - {agent}")
    
    train(agent_list, original_selection_masks, valid_action_index, full_action_space, n_episodes=3000, max_step=300, expected_flag=1, eps_start=1.0, eps_end=0.1, eps_decay=0.999)


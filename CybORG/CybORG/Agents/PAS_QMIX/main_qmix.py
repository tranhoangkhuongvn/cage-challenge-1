from tqdm import tqdm
from CybORG.Agents.PAS_QMIX.agents import Dueling_Agent
from CybORG.Agents.PAS_QMIX.networks import DuelingQNetwork, ParamNetwork


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 


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
    



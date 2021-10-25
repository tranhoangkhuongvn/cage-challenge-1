import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 


from CybORG import CybORG
import inspect

from CybORG.Agents import TestAgent
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper

class DuelingNetwork(nn.Module): 

    def __init__(self, obs, action_space, hidden_1=512, hidden_2=256): 

        super().__init__()
        self.obs = obs
        self.action_space = action_space
        self.model = nn.Sequential(nn.Linear(obs, hidden_1), 
                                   nn.ReLU(), 
                                   nn.Linear(hidden_1, hidden_2), 
                                   nn.ReLU())

        self.value_head = nn.Linear(hidden_2, 1)

        
        self.adv_head = nn.Linear(hidden_2, action_space)

    def forward(self, x): 

        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1,1)
        return q_val


class BranchingQNetwork(nn.Module):

    # TODO:
    # 1. Add in parameters init method
    def __init__(self, obs_space, action_space, hidden_1=512, hidden_2=256): 

        super().__init__()
        self.obs_space = obs_space
        self.obs_size = obs_space.shape[0]
        self.action_space = action_space
        

        self.model = nn.Sequential(nn.Linear(self.obs_size, hidden_1), 
                                   nn.ReLU(),
                                   nn.Linear(hidden_1, hidden_2), 
                                   nn.ReLU())

        self.value_head = nn.Linear(hidden_2, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(hidden_2, i) for i in self.action_space])
            

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)
        
        out = self.model(x)
        value = self.value_head(out)

        actions = [x(out) for x in self.adv_heads]
        
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1,1)
            actions[i] += value
            actions[i] = actions[i].squeeze()

        return actions

if __name__ == '__main__':
    # Test branching network
    print(torch.cuda.get_device_name())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    #scenario_name = "Scenario1"
    scenario_name = "scenario_12_hosts_2flag"
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario_name}.yaml'

    agent_name = 'Red'
    cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))

    # cyborg = OpenAIGymWrapper(agent_name=agent_name, env = CybORG(path, 'sim'))
    # cyborg = CybORG(path, "sim")
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=CybORG(path, 'sim'))
    observation = cyborg.reset(agent=agent_name)
    action_space = cyborg.get_action_space(agent_name) # 8, 3, 4, 1, 9, 2, 139, 9, 8, 1, 4
    
    print(action_space)
    print(observation.shape)

    
    mynet = BranchingQNetwork(observation, action_space).to(device)
    print(mynet)
    print(next(mynet.parameters()).device)


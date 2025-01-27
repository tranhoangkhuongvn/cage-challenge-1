from numpy.lib.function_base import select
from tqdm import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

from torch.utils.tensorboard import SummaryWriter

import numpy as np 
import gym 
import random
from tqdm import tqdm

from collections import namedtuple, deque

from network import DuelingNetwork, BranchingQNetwork
from utils import TensorEnv, ExperienceReplayMemory, AgentConfig, BranchingTensorEnv
import utils


from CybORG import CybORG
import inspect

from CybORG.Agents import TestAgent
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table
import pdb


BUFFER_SIZE = int(1000000)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LR = 1e-4
UPDATE_EVERY = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device used: ', device)
seed_list = [0, 42, 500, 1000]


class ReplayBuffer:
    """
    Fixed size buffer to store experience tuples
    """
    def __init__(self, action_space, buffer_size, batch_size, seed=42, device="cpu"):
        """
        Params:
            action_size (int): dimension of each action in the action space
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        #self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.action_space = action_space
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state","done"])
        self.seed = seed
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        # Multi-discrete action: a_t = [a1, a2, a3, a4]
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)


    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k =self.batch_size)

        # states: [batch_size, state_size] for example: [64, 605]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)

        action_list = [[] for i in range(len(self.action_space))]
        for idx in range(len(self.action_space)):
            # action_list[idx]: (batch_size, 1) for example: (64, 1)
            action_list[idx] = torch.from_numpy(np.vstack([e.action[idx] for e in experiences if e is not None])).float().to(self.device)

        # reward: [batch_size, 1] for example: [64, 1]
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        # next_tates: [batch_size, state_size] for example: [64, 605]
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # dones: [batch_size, 1] for example: [64, 1]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, action_list, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class BranchingDQN_Agent(nn.Module): 

    def __init__(self, observation, action_space, hidden_1 = 512, hidden_2=256, seed=0, device="cpu"): 

        super().__init__()

        self.observation = observation
        self.action_space = action_space
        self.qnetwork_local = BranchingQNetwork(self.observation, self.action_space, hidden_1, hidden_2).to(device)
        self.qnetwork_target = BranchingQNetwork(self.observation, self.action_space, hidden_1, hidden_2).to(device)

        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.device = device
        # Replay memory
        self.memory = ReplayBuffer(self.action_space, BUFFER_SIZE, BATCH_SIZE, seed, device=device)
		#initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.tau = TAU

    
    def step(self, state, action, reward, next_state, done):
		# Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
		# learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        #print(self.t_step, len(self.memory))
        if self.t_step == 0:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                
                experiences = self.memory.sample()
                # sample a batch of 64 (s,a,r,s',d)
                self.learn(experiences, GAMMA)

    
    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples

        Params:
            experiences: (Tuple[torch.Variable]): tuple of (s,a,r,s',done) tuples
            action in experience: a_t = (a1, a2, a3, a4)
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
        # reshape the actions:
        
        #actions = torch.stack(actions).transpose(0,1).long() # (batch, action_size, 1)
        
        # Calculate list of q_value for each action, for example: [(64, 6), (64, 5), (64, 13), (64, 13)]
        Q_expected = self.qnetwork_local(states) 
        #Q_expected = torch.stack(Q_expected)
        for i in range(len(self.action_space)):
            Q_expected[i] = Q_expected[i].gather(1, actions[i].long())
        
        
        # Q_expected = [ (64, 1), (64, 1), (64, 1), (64, 1)]
        # ==> (64, 4, 1) ==> (64, 4)
        Q_expected = torch.stack(Q_expected).transpose(0, 1) # (batch, num_action_dim, 1)
        Q_expected = Q_expected.squeeze(-1) # (batch, num_action_dim)

        
        # Get max predicted Q values (for next states) from target model
        # List of Q(s', a'): [(64, 6), (64, 5), (64, 13), (64, 13)]
        
        Q_targets_next = self.qnetwork_target(next_states)
        # Pick the max over a' for each Q(s', a')
        for i in range(len(self.action_space)):
            Q_targets_next[i] = Q_targets_next[i].max(-1, keepdim=True)[0]
        
        # Q_targets_next = [(64, 1), (64, 1), (64, 1), (64, 1)]
        # ==> (64, 4, 1)
        Q_targets_next = torch.stack(Q_targets_next).transpose(0, 1)

        # Approach #1: calculate a single target value across the action dimensions
        #Q_targets_next = Q_targets_next#.squeeze(-1) # (batch, action_size)
        # compute Q targets for current states
        #Q_targets = rewards + (gamma * Q_targets_next.mean(1) * (1 - dones))
        
        #compute loss
        #loss = F.mse_loss(Q_expected, Q_targets.repeat(1, len(self.action_space)))

        # Approach #2: use a different target value for each action dimension
        Q_targets_next = Q_targets_next.squeeze() # (64, 4, 1) ==> (64, 4)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #loss = F.mse_loss(Q_expected, Q_targets)
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        #Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        #nn.utils.clip_grad_norm_(list(self.qnetwork_local.parameters()), 2)

        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

	
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters
        Q_target = tau*Q_local + (1-tau)*Q_target

        Params:
            local_model (Pytorch model): weights will be copied from target_model (Pytorch model)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

    
    
    def act(self, state, selection_masks, eps=0.):
        # TODO: take into account selection masks
        #state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state = state.float().to(device)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        # Only select actions in the selection_masks

        
        
        for idx, action_value in enumerate(action_values):
            valid_action_idx = torch.tensor([True for _ in range(action_value.size(0))]) # [F, F, F, F, F, F]
            try:
                valid_action_idx[selection_masks[idx]] = False # [True, False, False, True, True, False]
                action_value[valid_action_idx] = float('-inf')
            except:
                #import pdb; pdb.set_trace()
                print(valid_action_idx)
                print(selection_masks[idx])
                print(idx)
                import pdb; pdb.set_trace()
                

        if random.random() > eps:
            #print("greedy")
            #return np.argmax(action_values.cpu().data.numpy())
            # TODO: why dont we use argmax?
            #actions = [int(x.max()[1]) for x in action_values]
            #actions = [x.argmax().item() for x in action_values]
            #import pdb; pdb.set_trace()
            actions = [argmax_rand(x.cpu().numpy()) for x in action_values]
            return actions
        else:
            #print("random")
            
            actions = []
            for action_indx in selection_masks:
                if len(action_indx) != 0:
                    actions.append(np.random.choice(action_indx))
                else:
                    actions.append(0)
            
            #return [np.random.randint(0, i) for i in self.action_space]
            return actions

    def get_action(self, x): 

        with torch.no_grad(): 
            # a = self.q(x).max(1)[1]
            out = self.q(x).squeeze(0) # remove first dim which was 1
            #print(f"out: {out.shape} - {out}")
            action = torch.argmax(out, dim = 1)
        return action.numpy()



def is_diff(s1, s2):
    s1 = np.array(s1)
    s2 = np.array(s2)

    diff = s1 - s2
    if sum(diff) == 0:
        return False
    
    return True


def numpy_to_torch(np_arr):
    assert isinstance(np_arr, np.ndarray), "input must be a numpy array"
    if isinstance(np_arr, torch.Tensor):
        return np_arr
    return torch.tensor(np_arr, dtype=torch.float64) 


def argmax_rand(arr):
	# np.argmax with random tie breaking
	return np.random.choice(np.flatnonzero(np.isclose(arr, np.max(arr), atol=1e-3)))

"""
Example of invalid action:
mask [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1], [0, 1, 2], [7], [1], [66], [0, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 5], [0, 2]]

action space: [8, 3, 4, 9, 2, 139, 15, 8, 4]
action taken: <class 'list'> - [1, 1, 2, 7, 1, 93, 10, 5, 2]
reward: -0.1

"""
def train(rl_agent, original_selection_masks, valid_action_index, full_action_space, n_episodes, max_step, expected_flag = 1, eps_start=1.0, eps_end=0.1, eps_decay=0.99):
    stats = utils.Stats(num_episodes=n_episodes, continuous=True)
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
        
        
        #action_space = cyborg.get_action_space(agent_name)
        for t in range(max_step):
            
            observation = numpy_to_torch(observation)
            
            actions = np.array([0 for _ in range(len(full_action_space))])
            
            action_mask = [selection_masks[i] for i in valid_action_index]
            
            rl_actions = rl_agent.act(observation, action_mask, eps)

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
                    rl_agent.step(observation, actions[valid_action_index], reward, next_observation, done)
                    print("Capture flat at:", t)
                    
                    break
            
            step = t
            
            rl_agent.step(observation, actions[valid_action_index], reward, next_observation, done)
            observation = next_observation
            score += reward
        
        if not done:
            print("Failed at step:", step)    
        scores_window.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {} Eps {}\tAverage Score: {:.2f}'.format(i_episode, eps, np.mean(scores_window)), end="")

def detect_host(host_list, current_host):
    if len(host_list) != len(current_host):
        for h in host_list:
            if h not in current_host:
                current_host.add(h)

        return current_host, True
    
    return current_host, False


if __name__ == '__main__':
    # Setup summary writer
    import time
    seed = 42
    experiment_name = f"cyborg__{seed}__{int(time.time())}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    writer = SummaryWriter(f"runs/{experiment_name}")
    
    

    # Test branching dqn agent
    from pprint import pprint
    print(torch.cuda.get_device_name())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    #scenario_name = "Scenario1"
    scenario_name = "scenario_12_hosts_2flag"
    
    print(scenario_name)
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario_name}.yaml'

    agent_name = 'Red'
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))

    
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=CybORG(path, 'sim'))
    #cyborg = CybORG(path, "sim")
    
    

    print("full")
    cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=CybORG(path, 'sim'))
    #cyborg = CybORG(path, "sim")
    
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
    test_bdqn_agent = BranchingDQN_Agent(obs_torch, valid_action_space, hidden_1=1024, hidden_2=1024, device=device)

    train(test_bdqn_agent, original_selection_masks, valid_action_index, full_action_space, n_episodes=10000, max_step=150, expected_flag=1, eps_start=1.0, eps_end=0.1, eps_decay=0.9999)

    


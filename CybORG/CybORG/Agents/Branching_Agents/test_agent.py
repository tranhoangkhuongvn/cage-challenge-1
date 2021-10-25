from tqdm import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

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


BUFFER_SIZE = int(300000)
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
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)


    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k =self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)

        action_list = [[] for i in range(len(self.action_space))]
        for idx in range(len(self.action_space)):
            action_list[idx] = torch.from_numpy(np.vstack([e.action[idx] for e in experiences if e is not None])).float().to(self.device)

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, action_list, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class BranchingDQN_Agent(nn.Module): 

    def __init__(self, observation, action_space, seed=0, device="cpu"): 

        super().__init__()

        self.observation = observation
        self.action_space = action_space
        self.qnetwork_local = BranchingQNetwork(self.observation, self.action_space).to(device)
        self.qnetwork_target = BranchingQNetwork(self.observation, self.action_space).to(device)

        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.device = device
        # Replay memory
        self.memory = ReplayBuffer(self.action_space, BUFFER_SIZE, BATCH_SIZE, seed, device=device)
		#initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


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
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
        # reshape the actions:
        
        #actions = torch.stack(actions).transpose(0,1).long() # (batch, action_size, 1)
        
        Q_expected = self.qnetwork_local(states) # list of q_value for each action
        #Q_expected = torch.stack(Q_expected)
        for i in range(len(self.action_space)):
            Q_expected[i] = Q_expected[i].gather(1, actions[i].long())
        
        Q_expected = torch.stack(Q_expected).transpose(0, 1) # (batch, action_size, 1)
        Q_expected = Q_expected.squeeze(-1) # (batch, action_size)

        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states)
        for i in range(len(self.action_space)):
            Q_targets_next[i] = Q_targets_next[i].max(-1, keepdim=True)[0]
        
        Q_targets_next = torch.stack(Q_targets_next).transpose(0, 1)
        Q_targets_next = Q_targets_next#.squeeze(-1) # (batch, action_size)
        # compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next.mean(1) * (1 - dones))
        
        #compute loss
        loss = F.mse_loss(Q_expected, Q_targets.repeat(1, len(self.action_space)))

        #Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

	
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

    
    
    def act(self, state, eps=0.):
        # TODO: take into account selection masks
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            #print("greedy")
            #return np.argmax(action_values.cpu().data.numpy())
            # TODO: why dont we use argmax?
            actions = [int(x.max(1)[1]) for x in action_values]
            return actions
        else:
            #print("random")
            return [np.random.randint(0, i) for i in self.action_space]


    def get_action(self, x): 

        with torch.no_grad(): 
            # a = self.q(x).max(1)[1]
            out = self.q(x).squeeze(0) # remove first dim which was 1
            #print(f"out: {out.shape} - {out}")
            action = torch.argmax(out, dim = 1)
        return action.numpy()


    def update_policy(self, adam, memory, params): 

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params.batch_size)

        states = torch.tensor(b_states).float() # batch of states [128, 24]
        actions = torch.tensor(b_actions).long().reshape(states.shape[0],-1,1)
        rewards = torch.tensor(b_rewards).float().reshape(-1,1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1,1)

        qvals = self.q(states) # [128, 4, 6]
        
        # [128, 4]: q_values for selected actions
        current_q_values = self.q(states).gather(2, actions).squeeze(-1)
        print(f"current_qvals: {current_q_values.shape}")
        
        with torch.no_grad():
            argmax = torch.argmax(self.q(next_states), dim = 2)
            print(f"argmax: {argmax.shape}")
            max_next_q_vals = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            print(f"max_next_q_vals: {max_next_q_vals.shape}")
            # calculate the avg across action dimension, keep 1 avg values
            max_next_q_vals = max_next_q_vals.mean(1, keepdim = True)
            print("-----------------")
            print(f"max_next_q_vals: {max_next_q_vals.shape}")

        expected_q_vals = rewards + max_next_q_vals*0.99*masks
        # print(expected_q_vals[:5])

        print(f"expected_q_vals: {expected_q_vals.shape}")
        print(f"current_q_values: {current_q_values.shape}")
        loss = F.mse_loss(expected_q_vals, current_q_values)
        
        # input(loss)

        # print('\n'*5)
        
        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters(): 
            p.grad.data.clamp_(-1.,1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0: 
            self.update_counter = 0 
            self.target.load_state_dict(self.q.state_dict())


def is_diff(s1, s2):
    s1 = np.array(s1)
    s2 = np.array(s2)

    diff = s1 - s2
    if sum(diff) == 0:
        return False
    
    return True


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
def train(rl_agent, n_episodes, max_step, eps_start=1.0, eps_end=0.1, eps_decay=0.99):
    stats = utils.Stats(num_episodes=n_episodes, continuous=True)
    scores = []
    steps = []
    scores_window = deque(maxlen=100)
    
    eps = eps_start
    
    for i_episode in range(n_episodes):
        print("\rEpisode: {}".format(i_episode))
        score = 0
        agent_name = 'Red'
        #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))
        observation = cyborg.reset(agent=agent_name)
        #action_space = cyborg.get_action_space(agent_name)
        for t in range(max_step):
            actions = rl_agent.act(observation, eps)
            next_observation, reward, done, info = cyborg.step(action=actions)

            rl_agent.step(observation, actions, reward, next_observation, done)
            observation = next_observation
            score += reward
            if done:
                print("Done:", t)
                break
        scores_window.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {} Eps {}\tAverage Score: {:.2f}'.format(i_episode, eps, np.mean(scores_window)), end="")






if __name__ == '__main__':
    # Test branching dqn agent
    from pprint import pprint
    print(torch.cuda.get_device_name())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    scenario_name = "Scenario1"
    #scenario_name = "scenario_12_hosts_2flag"
    
    print(scenario_name)
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario_name}.yaml'

    agent_name = 'Red'
    cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))

    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
    #cyborg = OpenAIGymWrapper(agent_name=agent_name, env = CybORG(path, 'sim'))
    #cyborg = CybORG(path, "sim")
    
    observation = cyborg.reset(agent=agent_name)
    action_space = cyborg.get_action_space(agent_name) # 8, 3, 4, 1, 9, 2, 139, 9, 8, 1, 4
    
    
    print(action_space)
    print(observation.shape)

    full_obs = cyborg.env.reset(agent=agent_name)

    print(type(full_obs.selection_masks))
    print(full_obs.selection_masks)
    exit(0)
    # for k,v in action_space.items():
    #     print(k)
    #     print(v)

    # for k, v in observation.observation.items():
    #     print(k)
    #     print(v)
    # #print(observation.observation)
    
    selection_masks = [[i for i in range(a)] for a in action_space]
    print(selection_masks)

    test_bdqn_agent = BranchingDQN_Agent(observation, action_space, device=device)

    #train(test_bdqn_agent, n_episodes=3000, max_step=1500, eps_start=1.0, eps_end=0.1, eps_decay=0.999)

    #print(len(test_bdqn_agent.memory))

    accum_reward = 0

    true_state = cyborg.get_agent_state('True')
            

    import pdb; 
    true_table = true_obs_to_table(true_state, cyborg)
    print(true_table)
    
    print(76*'-')
    pdb.set_trace()
    for i in range(5000):
        
        #import pdb; pdb.set_trace()
        actions = test_bdqn_agent.act(observation, eps=0.5)
        print(f"action: {actions}")
        #execute_action = []
        

        # for indx, action in enumerate(actions):
        #     if len(selection_masks[indx]) == 0:
        #         execute_action.append(None)
        #     else:
        #         execute_action.append(action)
        # print(f"execute action: {execute_action}")
            
        next_observation, r, done, info = cyborg.step(action=actions)

        
        #print("is diff:", is_diff(observation, next_observation))
        print(f"reward: {r}")
        accum_reward += r
        print(f"accum_reward: {accum_reward}")

        if done:
            observation = cyborg.reset(agent=agent_name)
            print("Capture the flag at:", i)

        # if r > 0:
        #     print("Pos reward at:", i)
        #     true_state = cyborg.get_agent_state('True')
            

        #     #pdb.set_trace()
        #     true_table = true_obs_to_table(true_state, cyborg)
        #     print(true_table)
            
        #     print(76*'-')
        #     #break

        # if r < 0:
        #     print("Invalid action")
        #     #break
        action_space = info.get('action_space')
        print(f"action space: {action_space}")
        print(info["selection_masks"])
        selection_masks = info["selection_masks"]
        observation = next_observation
    



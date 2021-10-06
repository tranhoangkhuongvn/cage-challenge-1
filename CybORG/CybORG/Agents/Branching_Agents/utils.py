import numpy as np 
import gym 
import torch 
import random
from argparse import ArgumentParser 
import os 
import pandas as pd 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d



class Stats():
	def __init__(self, num_episodes=20000, num_states = 6, log_dir='./', continuous=False):
		self.episode_rewards = np.zeros(num_episodes)
		self.episode_lengths = np.zeros(num_episodes)
		if not continuous:
			self.visitation_count = np.zeros((num_states, num_episodes))
			self.target_count = np.zeros((num_states, num_episodes))
		self.log_dir = log_dir	

	def log_data(self, file_name):
		save_name = self.log_dir + file_name
		np.savez(save_name, reward=self.episode_rewards, step=self.episode_lengths)


def plot_rewards(ax, episodes_ydata, smoothing_window = 100, label="",c='b', alpha=0.5):
	#smoothing_window = 100

	overall_stats_q_learning = []
	for trialdata in episodes_ydata:
		overall_stats_q_learning.append(pd.Series(trialdata.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean())
		#overall_stats_q_learning.append(pd.Series(trialdata.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
	m_stats_q_learning = np.mean(overall_stats_q_learning, axis=0)
	std_stats_q_learning = np.std(overall_stats_q_learning, axis=0)

	ax.plot(range(len(m_stats_q_learning)), m_stats_q_learning, label=label, c=c)
	ax.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=alpha, edgecolor=c, facecolor=c)
	#ax.set_ylabel('Score')
	#ax.set_xlabel('Episode #')
	#ax.grid()


def plot_steps(ax, episodes_ydata, smoothing_window = 100, label="",c='g',alpha=0.5):
	#smoothing_window = 100

	overall_stats_q_learning = []
	for trialdata in episodes_ydata:
		overall_stats_q_learning.append(pd.Series(trialdata.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean())
		#overall_stats_q_learning.append(pd.Series(trialdata.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
	m_stats_q_learning = np.mean(overall_stats_q_learning, axis=0)
	std_stats_q_learning = np.std(overall_stats_q_learning, axis=0)

	ax.plot(range(len(m_stats_q_learning)), m_stats_q_learning, label=label, c=c)
	ax.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=alpha, edgecolor=c, facecolor=c)
	#ax.set_ylabel('Steps')
	#ax.set_xlabel('Episode #')
	#ax.grid()

def plot_visitation_counts(episodes_ydata, smoothing_window = 1000, c=['b', 'g', 'r', 'y', 'k', 'c'], num_states = None):

    if not num_states: 
        num_states = len(episodes_ydata[0].visitation_count)

    overall_stats_q_learning = [[] for i in range(num_states)]
    for trialdata in episodes_ydata:
        for state in range(num_states):
            overall_stats_q_learning[state].append(pd.Series(trialdata.visitation_count[state]).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    
    for state in range(num_states):
        m_stats_q_learning = np.mean(overall_stats_q_learning[state], axis=0)
        std_stats_q_learning = np.std(overall_stats_q_learning[state], axis=0)

        plt.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c[state])
        plt.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c[state], facecolor=c[state])

def plot_target_counts(episodes_ydata, smoothing_window = 1000, c=['b', 'g', 'r', 'y', 'k', 'c']):

    num_states = len(episodes_ydata[0].target_count)

    overall_stats_q_learning = [[] for i in range(num_states)]
    for trialdata in episodes_ydata:
        for state in range(num_states):
            overall_stats_q_learning[state].append(pd.Series(trialdata.target_count[state]).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    
    for state in range(num_states):
        m_stats_q_learning = np.mean(overall_stats_q_learning[state], axis=0)
        std_stats_q_learning = np.std(overall_stats_q_learning[state], axis=0)

        plt.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c[state])
        plt.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c[state], facecolor=c[state])

def plot_q_values(model, observation_space, action_space):

    res = 100

    test_observations = np.linspace(observation_space.low, observation_space.high, res)
    
    print((action_space.n, res))
    q_values = np.zeros((action_space.n, res))

    for action in range(action_space.n):
        for obs in range(res):
            q_values[action, obs] = model.predict(test_observations[obs])[0, action]

        plt.plot(test_observations, q_values[action])

def arguments(): 

    parser = ArgumentParser()
    parser.add_argument('--env', default = 'BipedalWalker-v3')

    return parser.parse_args()


def save(agent, rewards, args): 

    path = './runs/{}/'.format(args.env)
    try: 
        os.makedirs(path)
    except: 
        pass 

    torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict'))

    plt.cla()
    plt.plot(rewards, c = 'r', alpha = 0.3)
    plt.plot(gaussian_filter1d(rewards, sigma = 5), c = 'r', label = 'Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)





class AgentConfig:

    def __init__(self, 
                 epsilon_start = 1.,
                 epsilon_final = 0.01,
                 epsilon_decay = 8000,
                 gamma = 0.99, 
                 lr = 1e-4, 
                 target_net_update_freq = 1000, 
                 memory_size = 100000, 
                 batch_size = 128, 
                 learning_starts = 5000,
                 max_frames = 10000000): 

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_by_frame = lambda i: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * i / self.epsilon_decay)

        self.gamma =gamma
        self.lr =lr

        self.target_net_update_freq =target_net_update_freq
        self.memory_size =memory_size
        self.batch_size =batch_size

        self.learning_starts = learning_starts
        self.max_frames = max_frames


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])


        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


import torch

import collections
import random

class ReplayBuffer():
    def __init__(self,buffer_limit,action_space,device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_space = action_space
        self.device = device
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [[] for i in range(self.action_space)]

        for transition in mini_batch:
            state, actions,reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        actions_lst = [torch.tensor(x,dtype= torch.float).to(self.device) for x in actions_lst]
        return torch.tensor(state_lst, dtype=torch.float).to(self.device),\
               actions_lst ,torch.tensor(reward_lst).to(self.device),\
                torch.tensor(next_state_lst, dtype=torch.float).to(self.device),\
               torch.tensor(done_mask_lst).to(self.device)
    def size(self):
        return len(self.buffer)


class TensorEnv(gym.Wrapper): 

    def __init__(self, env_name): 

        super().__init__(gym.make(env_name))

    def process(self, x): 

        return torch.tensor(x).reshape(1,-1).float()

    def reset(self): 

        return self.process(super().reset())

    def step(self, a): 

        ns, r, done, infos = super().step(a)
        return self.process(ns), r, done, infos 


class BranchingTensorEnv(TensorEnv): 

    def __init__(self, env_name, n): 

        super().__init__(env_name)
        self.n = n 
        self.discretized = np.linspace(-1.,1., self.n)


    def step(self, a):

        action = np.array([self.discretized[aa] for aa in a])

        return super().step(action)

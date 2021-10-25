import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

from networks import DuelingQNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import argmax_rand
from replay_buffer import ReplayBufferSingleAgent

class Dueling_Agent():
    def __init__(self, state_size, action_size, args, seed=0, device="cpu"):
        """Initialize an Agent object
        Params
            ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed #random.seed(seed)
        self.device = args.device
        self.update_freq = args.update_freq
        self.gamma = args.gamma
        self.batch = args.batch
        self.tau = args.tau
        self.lr = args.lr

        #Q - network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size).to(self.device)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.lr)
        # Replay memory
        self.memory = ReplayBufferSingleAgent(action_size, args.buffer, args.batch, self.seed, device=self.device)
        #initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

	

    def act(self, state, action_mask, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state = state.float().to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()


        
        valid_action_idx = torch.tensor([True for _ in range(action_values.size(0))]) # [T, T, T, T, T, T]
            
        valid_action_idx[action_mask] = False # [True, False, False, True, True, False]
        action_values[valid_action_idx] = float('-inf')
            
                

        if random.random() > eps:
            #print("greedy")
            action = argmax_rand(action_values.cpu().data.numpy())
            return action
        else:
            #print("random")
            action = np.random.choice(action_mask)
            return action


    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples

        Params:
            experiences: (Tuple[torch.Variable]): tuple of (s,a,r,s',done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        #compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        #Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
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

import time
import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')


BUFFER_SIZE = int(300000)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LR = 1e-4
UPDATE_EVERY = 10

def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments StarCraft II")

    # environment
    parser.add_argument("--device", type=str, default="cpu", help="cpu or gpu")
    parser.add_argument("--buffer", type=int, default=1000000, help="replay buffer size")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--gamma", type=int, default=0.99, help="discount factor")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument("--tau", type=float, default=0.001, help="soft update parameter")
    parser.add_argument("--update_freq", type=int, default=10, help="network learning frequency")
    parser.add_argument("--hidden_dim_1", type=int, default=512, help="hidden dimension 1")
    parser.add_argument("--hidden_dim_2", type=int, default=256, help="hidden dimension 2")
    return parser.parse_args()
import torch
import torch.multiprocessing as mp
from PPO import PolicyNet, ValueNet
from torch.distributions import Categorical


class DPPO(mp.Process):
    def __init__(self, policy):
        super(DPPO, self).__init__()


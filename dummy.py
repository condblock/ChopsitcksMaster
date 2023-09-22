import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")
device = torch.device('cpu')

print(env.action_space.n)
# 상태 관측 횟수를 얻습니다.
print(env.reset())
print(env.action_space.sample())
state, info = env.reset()
print(state)
print(info)
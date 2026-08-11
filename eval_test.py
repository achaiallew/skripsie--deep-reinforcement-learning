#============================================================================
# Import Modules
#============================================================================
# Imports for Environment
import gymnasium as gym
import minigrid
from minigrid.wrappers import *

# Imports for Generate Random Numbers / Epsilon Decay
import random
import math
# Import for Performance Tracking
import time

# Import for Various Mathematical, Vector and Matrix Functions
import numpy as np

from os.path import exists

# Import for Hyperparameter Tuning
import optuna

# Import for Experience Replay Memory
from collections import namedtuple, deque

# Imports for Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Imports for Writing toTensorboard
from torch.utils.tensorboard import SummaryWriter

from deep_q_network import DQN

# Check for GPU Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


filename = 'dqn_trained.pth'
# Reload the Newly Trained Model
if exists(filename):
    policy_net = torch.load(filename, map_location=device, weights_only=False)
    policy_net.eval()
    print(f'Loaded trained model from {filename}')
else:
    raise FileNotFoundError(f'No saved model found at {filename}.')

print(policy_net.fc1.weight)
print(policy_net.fc2.weight)
print(policy_net.fc3.weight)
#============================================================================
# Import Modules
#============================================================================
# Import 'gymnasium' and 'minigrid' for our Environment
import gymnasium as gym
import minigrid
from minigrid.wrappers import *

# Import 'random' to Generate Random Numbers
import random

# Import 'numpy' for Various Mathematical, Vector and Matrix Functions
import numpy as np

from os.path import exists

# Import 'Pytorch' for Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# If GPU is to be Used Otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import 'namedtuple' and 'deque' for Experience Replay Memory
from collections import namedtuple, deque

#============================================================================
# Preprocessing Fuctions
#============================================================================
# Extract the OBJECT_IDX Information as a Matrix using Numpy Slicing and Reshaping
def extractObjectInformation2(observation):
    (rows, cols, x) = observation.shape
    tmp = np.reshape(observation,[rows*cols*x,1], 'F')[0:rows*cols]
    return np.reshape(tmp, [rows,cols],'C')

# Normalise the Input Observation so each element is a scalar value between [0,1]
def normalize(observation, max_value):
    return np.array(observation)/max_value

# Flatten the [7,7] Matrix into a [1,49] tensor
def flatten(observation):
    return torch.from_numpy(np.array(observation).flatten()).float().unsqueeze(0)

# Combine Preprocessing Fuctions
def preprocess(observation):
    return flatten(normalize(extractObjectInformation2(observation), 10.0))

#============================================================================
# SetUp the RL Agent
#============================================================================
# Gym Envrionment
env = gym.make('MiniGrid-Empty-8x8-v0')

# Use Wrapper so the Observation only contains the Grid Information
env = ImgObsWrapper(env)

obs, _ = env.reset()
print(extractObjectInformation2(obs))

# Variables
steps_done = 0

#============================================================================
# SetUp the HyperParameters
#============================================================================
# ---- MODEL HYPERPARAMETERS ----
numActions = 3                  # left, right, forward
inputSize = 49                  # size of flattened input state (7x7 matrix of tile IDs) 

# ---- TRAINING HYPERPARAMETERS ----
alpha = 0.0002                  # learning rate
episodes = 5000                 # total episodes for training
batchSize = 128                 # neural network batch size
target_update = 20000           # no. episodes bet. updating target network 

# ---- Q-LEARNING HYPERPARAMETERS ----
gamma = 0.90                    # discounting rate

# ---- EXPLORATION PARAMETERS for Epsilon Greedy Strategy ----
start_epsilon = 1.0             # exploration probability at start
stop_epsilon = 0.01             # minimum exploration probability
decay_rate = 20000              # exponential decay rate for exploration probability

# ---- MEMORY HYPERPARAMETERS ---- 
pretrain_length = batchSize     # no. experiences stored in memory on initialisation
memSize =  500000               # no. experiences the memory can keep - 500000

# ---- TESTING HYPERPARAMETERS ----
evalEpisodes = 1000             # no. episodes to be used for eval
train = True                    # True to train a model; False to eval prev trained agent 

#============================================================================
# Define and Create a Neural Network Model
#============================================================================
# Neural Network Model Definition
class DQN(nn.Module):

    def __init__(self, inputSize, numActions, hiddenLayerSize=(512, 256)):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenLayerSize[0])
        self.fc2 = nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1])
        self.fc3 = nn.Linear(hiddenLayerSize[1], numActions)

    ''' Called with either one element to determine next action, or a batch
        during optimisation. Returns tensor([[left0exp, right0exp]...])'''
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the Policy Network and Target Network
hiddenLayerSize = (128, 128)
policy_net = DQN(inputSize, numActions, hiddenLayerSize)
target_net = DQN(inputSize, numActions, hiddenLayerSize)

# Copy Weights of Policy Network to Target Network
target_net.load_state_dict(policy_net.state_dict())

# Set Target Netwrok to Eval Mode to not Update Parameters
target_net.eval()

# ---- SAVING AND LOADING OF NETWORKS ----
filename = 'test.pth'

# Saving a Model
torch.save(policy_net, filename)

# Loading a Model 
#loaded_model = torch.load(filename)


# ---- PERFORM FORWARD PASS OF POLICY NETWORK ----
# Reset the Environment
obs, _ = env.reset()

# Preprocess the Observation to obtain State
state = preprocess(obs)

# Apply State as Input to Policy Network
action_values = policy_net(state)

'''If want want to get the action that has the highest Q-value we use the 'max' function. 
   The result is a tuple where the first element is the value, and the second element is the index'''

# Text Best Action
print('action_values: ',action_values)
print('\nbest action: ', action_values.max(1))
a = action_values.max(1)[1]
print('\na: ', a)

#============================================================================
# Epsilon-Greedy Exploration
#============================================================================
def select_action(state):
    # Generate a Random Number
    r = random.random()

    # Calculate the Epsilon Threshold
    epsilon_thres = stop_epsilon + (start_epsilon - stop_epsilon)*math.exp(-1. * steps_done/decay_rate)

    # Compare Random Number to Epsilon Threshold
    if r > epsilon_thres:
        # Act Greedily Toward Q-Values of Policy Network given State
        # Don't Want to Gather Gradients as we are Only Generating Experience not Training the Network
        with torch.no_grad():
            '''t.max(1) will return largest column value of each row. 
            second column on max result is index of where max element was
            found, so we pick action with larger expected reward.'''    
            return policy_net(state).max(1)[1].unsqueeze(0)
    else:
        #Select Random Action with Equal Probability
        return torch.tensor([[random.randrange(numActions)]], device=device, dtype=torch.long)
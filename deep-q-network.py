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
writer = SummaryWriter()

# Check for GPU Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#============================================================================
# Preprocessing Functions
#============================================================================
# Extract Object_Idx Info Using Numpy Slicing and Reshaping
def extractObjectInformation2(observation):
    (rows, cols, x) = observation.shape
    tmp = np.reshape(observation, [rows*cols*x, 1], 'F')[0:rows*cols]
    return np.reshape(tmp, [rows, cols], 'C')

# Normalise the Input Observation: [0,1]
def normalize(observation, max_value):
    return np.array(observation)/max_value

# Flatten the [7,7] Matrix into a [1,49] Tensor
def flatten(observation):
    return torch.from_numpy(np.array(observation).flatten()).float().unsqueeze(0)

# Combine Preprocessing Functions
def preprocess(observation):
    return flatten(normalize(extractObjectInformation2(observation), 10.0))

#============================================================================
# Configuration
#============================================================================
# Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
# Use Wrapper so the Observation only contains the Grid Information
env = ImgObsWrapper(env)

# Configure Max Steps
max_steps = env.max_steps

#============================================================================
# SetUp the HyperParameters
#============================================================================
# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  # left, right, forward
input_size = 49                  # size of flattened input state (7x7 matrix of tile IDs)
steps_done = 0

# ---- TRAINING HYPERPARAMETERS ----
alpha = 0.0002                   # learning rate
episodes = 3000                  # total episodes for training (per assignment spec)
batch_size = 128                 # neural network batch size
target_update = 20000            # no. steps bet. updating target network

# ---- Q-LEARNING HYPERPARAMETERS ----
gamma = 0.90                     # discounting rate

# ---- EXPLORATION PARAMETERS for Epsilon Greedy Strategy ----
start_epsilon = 1.0              # exploration probability at start
stop_epsilon = 0.01              # minimum exploration probability
decay_rate = 20000               # exponential decay rate for exploration probability

# ---- MEMORY HYPERPARAMETERS ----
pretrain_length = batch_size     # no. experiences stored in memory on initialisation
mem_size = 500000                # no. experiences the memory can keep

# ---- TESTING HYPERPARAMETERS ----
eval_episodes = 1000              # no. episodes to be used for eval
train = True                      # True to train a model; False to eval prev trained agent
filename = 'dqn_trained.pth'

#============================================================================
# Define and Create a Neural Network Model
#============================================================================
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
policy_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)
target_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)

# Copy Weights of Policy Network to Target Network
target_net.load_state_dict(policy_net.state_dict())

# Set Target Network to Eval Mode to not Update Parameters
target_net.eval()

#============================================================================
# Optimiser (defined ONCE here)
#============================================================================
optimiser = optim.Adam(policy_net.parameters(), lr=alpha)

#============================================================================
# Experience Replay Memory SetUp
#============================================================================
Transition = namedtuple('Transition', ('currentState', 'action', 'nextState', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a Transition'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Instantiate Memory
memory = ReplayMemory(mem_size)

#============================================================================
# Epsilon-Greedy Exploration
#============================================================================
def select_action(state, greedy=False):
    '''Select an action using epsilon-greedy exploration.
       If greedy=True, always act greedily (used for evaluation).'''
    global steps_done

    if greedy:
        with torch.no_grad():
            return policy_net(state).max(1)[1].unsqueeze(0)

    # Generate a Random Number
    r = random.random()

    # Calculate the Epsilon Threshold
    epsilon_thres = stop_epsilon + (start_epsilon - stop_epsilon) * math.exp(-1. * steps_done / decay_rate)

    # Compare Random Number to Epsilon Threshold
    if r > epsilon_thres:
        # Act Greedily Toward Q-Values of Policy Network given State
        with torch.no_grad():
            '''t.max(1) will return largest column value of each row.
            second column on max result is index of where max element was
            found, so we pick action with larger expected reward.'''
            return policy_net(state).max(1)[1].unsqueeze(0)
    else:
        # Select Random Action with Equal Probability
        return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)

#============================================================================
# Optimise Model
#============================================================================
def optimise_model():

    # Check if Replay memory has Stored Enough Experience
    if len(memory) < batch_size:
        return

    # ---- SAMPLE MINI-BATCH ----
    experience = memory.sample(batch_size)

    # Transpose the Batch (Convert Batch-Array of Experience to Experience of Batch-Arrays)
    batch = Transition(*zip(*experience))

    # ---- CALC ACTION-VALUE PREDICTED BY POLICY NETWORK ----
    state_batch = torch.cat(batch.currentState)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    '''Calculate the Action-Values for each State in Batch,
        and then gather the Q-Values for Action Associated with Specific State'''
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # ---- CALC TD-TARGET ESTIMATED BY TARGET NETWORK ----
    nf_next_states = torch.cat([s for s in batch.nextState if s is not None])

    # Initialise the next_state values to be zeroes
    next_state_values = torch.zeros(batch_size, device=device)

    # Compute a Mask of Non-Final States and Concatenate the Batch Elements
    nf_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.nextState)),
                            device=device, dtype=torch.bool)

    # Calculate the Estimated 'next_state' values for Non-Final States
    next_state_values[nf_mask] = target_net(nf_next_states).max(1)[0].detach()

    # Compute the Expected Q-Values (TD-Target)
    TDtargets = (next_state_values * gamma) + reward_batch

    # ---- CALC LOSS using MINIMUM-SQUARED-ERROR CRITERION ----
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, TDtargets.unsqueeze(1))

    # TD-error: how far off the policy network's estimate was from the target
    TDerrors = TDtargets.unsqueeze(1) - state_action_values

    # ---- MAKE GRADIENT DESCENT STEP TO MINIMISE LOSS ----
    optimiser.zero_grad()
    loss.backward()

    # Clamp the Gradients in Policy Network to Range [-1, 1]
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimiser.step()

    # Log Loss to Tensorboard
    writer.add_scalar('Loss', loss.item(), steps_done)
    writer.add_scalar('TDError', TDerrors.abs().mean().item(), steps_done)

#============================================================================
# Pre-fill Replay Memory with Random Experience
#============================================================================
def prefill_memory(length):
    print('Pre-filling replay memory...')
    obs, _ = env.reset()
    state = preprocess(obs)

    for i in range(length):
        # Take a fully random action (no network involved yet)
        action = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
        a = action.item()

        obs, reward, done, truncated, info = env.step(a)
        reward_tensor = torch.tensor([reward], device=device)

        if done or truncated:
            next_state = None
        else:
            next_state = preprocess(obs)

        memory.push(state, action, next_state, reward_tensor)

        if done or truncated:
            obs, _ = env.reset()
            state = preprocess(obs)
        else:
            state = next_state

    print(f'Replay memory pre-filled with {len(memory)} experiences.')

#============================================================================
# Main Training Loop
#============================================================================
print('Start training...')

if train:
    # Warm-start the replay buffer before training begins
    prefill_memory(pretrain_length)

    for e in range(episodes):
        # Reset the Environment
        obs, _ = env.reset()
        # Preprocess the Observation to Obtain State
        state = preprocess(obs)

        for s in range(0, max_steps):
            # Perform Epsilon-Greedy Action Selection
            action = select_action(state)
            a = action.item()

            # Perform the Action in Environment
            obs, reward, done, truncated, info = env.step(a)
            reward_tensor = torch.tensor([reward], device=device)
            steps_done += 1

            # Store Transition
            if done or truncated:
                next_state = None
            else:
                next_state = preprocess(obs)

            # Store Transition in Experience Replay Memory
            memory.push(state, action, next_state, reward_tensor)

            # Move to Next State
            state = next_state

            # Train Model
            optimise_model()

            # Periodically Update Target Network
            if steps_done % target_update == 0:
                print(f'Updating target network at step {steps_done}')
                target_net.load_state_dict(policy_net.state_dict())

            # Log Reward / Episode Length to Tensorboard
            if done or truncated:
                writer.add_scalar('Reward', reward, e)
                writer.add_scalar('EpisodeLength', s, e)
                break

        # Periodically Track Episode and Step Progress
        if e % 100 == 0:
            print(f'Episode {e}/{episodes} | Steps Done: {steps_done}')

    print('Done training...')

    # Save the Trained Model
    torch.save(policy_net, filename)

else:
    # Load a Previously Trained Model
    if exists(filename):
        policy_net = torch.load(filename, map_location=device)
        policy_net.eval()
        print(f'Loaded trained model from {filename}')
    else:
        raise FileNotFoundError(f'No saved model found at {filename}. Set train=True first.')

#============================================================================
# Evaluate Agent Performance
#============================================================================
print('Starting Evaluation...')
eval_counter = 0.0
total_steps = 0.0
total_reward = 0.0

for e in range(eval_episodes):
    # Initialise the Environment and State
    currentObs, _ = env.reset()
    currentState = preprocess(currentObs)

    # Main RL Loop
    for i in range(0, max_steps):
        # Always act greedily during evaluation (no exploration)
        action = select_action(currentState, greedy=True)
        a = action.item()

        obs, reward, done, truncated, info = env.step(a)

        if done or truncated:
            nextState = None
        else:
            nextState = preprocess(obs)

        if done or truncated:
            total_reward += reward
            total_steps += env.unwrapped.step_count
            if done:
                print('Finished evaluation episode %d with reward %f, %d steps, reaching goal '
                      % (e, reward, env.unwrapped.step_count))
                eval_counter += 1
            if truncated:
                print('Failed evaluation episode %d with reward %f, %d steps'
                      % (e, reward, env.unwrapped.step_count))
            break

        currentState = nextState

# Print a Summary of the Evaluation Results
print('Completion rate %.2f with average reward %0.4f and average steps %0.2f'
      % (eval_counter/eval_episodes, total_reward/eval_episodes, total_steps/eval_episodes))

writer.close()
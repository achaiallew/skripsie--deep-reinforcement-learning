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
# Configure Max Steps
max_steps = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped.max_steps

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

# ---- EXPLORATION HYPERPARAMETERS for Epsilon Greedy Strategy ----
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
# Optimiser (defined ONCE here)
#============================================================================
optimiser = optim.Adam(policy_net.parameters(), lr=alpha)

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
# Deep Q-Learning Function
#============================================================================
def deep_q_learning():

    # Declare Tracking Variables
    min_steps = 1000
    global steps_done
    episode_rewards = []

    # Episode Loop
    for e in range(episodes):
        # Declare Epsiode Tracking Variables
        episode_steps = 0 
        episode_reward = 0.0

        # Reset the Environment
        obs, _ = env.reset()

        # Preprocess the Observation to Obtain State
        state = preprocess(obs)

        for s in range(0, max_steps):
            # Perform Epsilon-Greedy Action Selection
            action = select_action(state)
            a = action.item()

            # Perform the Action in Environment
            obs, reward, done, truncated, _ = env.step(a)
            reward = torch.tensor([reward], device=device)

            # Increment Step Counters
            episode_steps += 1
            steps_done += 1
            # Calculate Accumulated Rewards
            episode_reward += reward

            # Preprocess the Observation to Obtain Next State
            if done or truncated:
                next_state = None
            else:
                next_state = preprocess(obs)

            # Store Transition in Experience Replay Memory
            memory.push(state, action, next_state, reward )

            # Train Model
            optimise_model()

            # Move to Next State
            state = next_state

            # Periodically Update Target Network
            if steps_done % target_update == 0:
                print(f'Updating target network at step {steps_done}')
                target_net.load_state_dict(policy_net.state_dict())

            # Log Reward / Episode Length to Tensorboard
            if done or truncated:
                break

        # Periodically Track Episode and Step Progress
        #if e % 100 == 0:
            #print(f'Episode {e}/{episodes} | Steps Done: {steps_done}')

        # Epsiode Rewards
        episode_rewards.append(episode_reward)

        # Track the Minimum Steps Taken
        if (min_steps > episode_steps):
            min_steps = episode_steps   

        # Write to Tensorboard upon Completion
        if writer is not None:
            writer.add_scalar('Reward/train', reward, steps_done)
            writer.add_scalar('Steps/train', s, steps_done)

    return np.mean(episode_rewards[-100:]), min_steps

# #============================================================================
# # Tune Hyperparameters using Optuna
# #============================================================================
# def objective(trial):
#     # Tune Model Hyperparameters
#     alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
#     gamma = trial.suggest_float("gamma", 0.8, 0.999)
#     epsilon_start = trial.suggest_float("epsilon_start", 0.5, 1.0)
#     epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.9999)
#     epsilon_min = trial.suggest_float("epsilon_min", 0.001, 0.1)

#     # Declare Empty Rewards Array
#     rewards = []
#     # Start Tuning Time
#     start_time = time.time()

#     for seed in range(3): 
#         np.random.seed(seed)
#         random.seed(seed)

#         # Make the Gym Environment
#         env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
#         env = ImgObsWrapper(env) 

#         # Search the Algorithm
#         mean_reward,_ = deep_q_learning(
#             env, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min,
#             episodes=800, log_to_tb=False, writer=None) 
        
#         rewards.append(mean_reward)
#         env.close()

    
#     elapsed = time.time() - start_time
#     print(f'Trial Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')

#     # Return the Average of the Trial Rewards
#     return np.mean(rewards)

#============================================================================
# Run the Study
#============================================================================
if __name__ == "__main__":
        
    # Load a Previously Trained Model
    if exists(filename):
        policy_net = torch.load(filename, map_location=device)
        policy_net.eval()
        print(f'Loaded trained model from {filename}')
    else:
        #raise FileNotFoundError(f'No saved model found at {filename}.')
        print("No Saved Model")

    # Gym Environment
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
    env = ImgObsWrapper(env)
        
    # SetUp Tensorboard
    writer = SummaryWriter()

    # Start Training
    print('Start training...')
    start_time = time.time()

    # Train the Model
    _, final_steps = deep_q_learning()

    # Close the Environment
    env.close()

    # Done Training
    print('Done Training...')
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Training Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
    print(f'Final Steps Taken: {final_steps}')

    # Flush Remaining Data
    writer.flush()
    writer.close()

    # Save the Trained Model
    torch.save(policy_net, filename)





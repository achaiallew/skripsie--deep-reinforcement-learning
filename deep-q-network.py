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
# SetUp Tensorboard
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
# Environment Setup
#============================================================================
# Gym Envrionment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
max_steps = env.max_steps

# Use Wrapper so the Observation only contains the Grid Information
env = ImgObsWrapper(env)

#============================================================================
# SetUp the HyperParameters
#============================================================================
# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  # left, right, forward
input_size = 49                  # size of flattened input state (7x7 matrix of tile IDs) 
steps_done = 0

# ---- TRAINING HYPERPARAMETERS ----
alpha = 0.0002                  # learning rate
episodes = 5000                 # total episodes for training
batch_size = 128                 # neural network batch size
target_update = 20000           # no. episodes bet. updating target network 

# ---- Q-LEARNING HYPERPARAMETERS ----
gamma = 0.90                    # discounting rate

# ---- EXPLORATION PARAMETERS for Epsilon Greedy Strategy ----
start_epsilon = 1.0             # exploration probability at start
stop_epsilon = 0.01             # minimum exploration probability
decay_rate = 20000              # exponential decay rate for exploration probability

# ---- MEMORY HYPERPARAMETERS ---- 
pretrain_length = batch_size     # no. experiences stored in memory on initialisation
mem_size =  500000               # no. experiences the memory can keep - 500000

# ---- TESTING HYPERPARAMETERS ----
eval_episodes = 1000             # no. episodes to be used for eval
train = True                    # True to train a model; False to eval prev trained agent 
filename = 'dqn_trained.pth'

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
policy_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)
target_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)

# Copy Weights of Policy Network to Target Network
target_net.load_state_dict(policy_net.state_dict())

# Set Target Netwrok to Eval Mode to not Update Parameters
target_net.eval()

#============================================================================
# Optimiser (defined ONCE here)
#============================================================================
# Create the Optimiser
optimiser = optim.Adam(policy_net.parameters(), lr=alpha)

#============================================================================
# Experience Replay Memory SetUp
#============================================================================
# Experience Transition Tuple
Transition = namedtuple('Transition', ('currentState', 'action', 'nextState', 'reward'))

# Replay Memory Buffer
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
        return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)

#============================================================================
# Optimise Model
#============================================================================   
def optimise_model():

    # Check if Replay memory has Stored Enough Experience
    if len(memory) < batch_size:
        return

    # ---- SAMPLE MINI-BATCH ----
    # Sample a Mini-Batch of Experience from Replay Memory
    experience = memory.sample(batch_size)

    # Transpose the Batch (Convert Batch-Array of Experience to Experience of Batch-Arrays)
    batch = Transition(*zip(*experience))

    # ---- CALC ACTION-VALUE PREDICTED BY POLICY NETWORK ----
    # Extract the Array of States, Actions and Rewards in Mini-Batch
    state_batch = torch.cat(batch.currentState)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    '''Calculate the Action-Values for each State in Batch, 
        and then gather the Q-Values for Action Associated with Specific State'''
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # ---- CALC TD-TARGET ESTIMATED BY TARGET NETWORK ----
    # Extract the Non-Final Next States and Concatenate Them
    nf_next_states = torch.cat([s for s in batch.nextState
                                            if s is not None])

    # Initialise the next_state values to be zeroes
    next_state_values = torch.zeros(batch_size, device=device)

    # Compute a Mask of Non-Final States and Concatenate the Batch Elements
    ''' (a final state would've been the one after which the episode ended)
        This is just a tensor of boolean values, one for each experience in the mini-batch'''
    nf_mask = torch.tensor(tuple(map(lambda s : s is not None, 
                                        batch.nextState)), device=device, dtype=torch.bool)

    # Calculat the Estimated 'next_state' values for Non-Final States
    next_state_values[nf_mask] = target_net(nf_next_states).max(1)[0].detach()

    # Compute the Expected Q-Values (TD-Target)
    TDtargets = (next_state_values * gamma) + reward_batch

    # Compute TD-Errors
    TDerrors = TDtargets.unsqueeze(1) - state_action_values

    # ---- CALC LOSS using MINIMUM-SQUARED-ERROR CRITERION ----
    # Configure the MSELoss
    criterion = nn.MSELoss()

    # Calulate the Loss for the Mini-Batch
        # make sure to resize TDtargets to be same shape as state-action value tensor
    loss = criterion(state_action_values, TDtargets.unsqueeze(1))

    # ---- MAKE GRADIENT DESCENT STEP TO MINIMISE LOSS ----
    # Zero the Gradient
    optimiser.zero_grad()

    # Calculate the Gradients by Backpropagation from Loss Function
    loss.backward()

    # Clamp the Gradients in Policy Network to Range [-1, 1]
        # helps stabalise training
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1) 

    # Update the Network Parameters of the Policy Network uisng the Optimiser
    optimiser.step()

    # Log Loss to Tensorboard
    writer.add_scalar('Loss', loss.item(), steps_done)

#============================================================================
# Main Training Loop 
#============================================================================
# Start Training
print('Start training...')

# Train Model
if train:
    for e in range(episodes):
        # Reset the Environment
        obs, _ = env.reset()
        # Preprocess the Observation to Obtain State
        state = preprocess(obs)


        for s in range(0, max_steps):
            # Perform Epsilon-Greedy Action Election
            action = select_action(state)
            a = action.item()
            # Perform the Action in Environment (Next State and Reward)
            obs, reward, done, truncated, info = env.step(a)
            reward_tensor = torch.tensor([reward], device = device)
            steps_done += 1

            # Store Transition
            if done or truncated:
                next_state = None
            else:
                # Preprocess the Observation of Next State
                next_state = preprocess(obs)

            # Store Transition in Experience Replay Memory
            memory.push(state, action, next_state, reward_tensor)

            # Move to Next State
            state = next_state

            # Train Model
            optimise_model()
            
            # Periodically Update Target Network 
                # copying all weights and biases from the policy network
            if steps_done % target_update == 0:
                print(f'Updating target network at step {steps_done}')
                target_net.load_state_dict(policy_net.state_dict())

            # Log Reward to Tensorboard
            if done or truncated:
                writer.add_scalar('Reward', reward, e)
                break
        
        # Periodically Track Episode and Step Progress
        if e % 100 == 0:
            print(f'Episode {e}/{episodes} | Steps Done: {steps_done}')

    # Done Training
    print('Done training...')
            
    # Save the Trained Model
    torch.save(policy_net, filename)

# Load A Model
else:
    # Load
    loaded_model = torch.load(filename)
    policy_net.eval()

#============================================================================
# Evaluate Agent Performance
#============================================================================
# Evaluation Loop
print('Starting Evaluation...')
eval_counter = 0.0
total_steps = 0.0
total_reward = 0.0

stop_epsilon = 0.0


for e in range(eval_episodes):
    # Initialise the Environment and State
    currentObs, _ = env.reset()
    currentState = preprocess(currentObs)
   
    # Main RL Loop
    for i in range(0, max_steps):
        # Select and Perform an Action
        action = select_action(currentState)
        a = action.item()

        # take action 'a', receive reward 'reward', and observe next state 'obs'
        # 'done' indicate if the termination state was reached
        obs, reward, done, truncated, info = env.step(a)
        
        if (done or truncated):
            nextState = None
        else:
            # Observe Next State
            nextState = preprocess(obs)

        if (done or truncated):
            total_reward += reward
            total_steps += env.unwrapped.step_count
            if (done):
                print('Finished evaluation episode %d with reward %f,  %d steps, reaching goal ' % (e, reward, env.unwrapped.step_count))
                eval_counter += 1
            if (truncated):
                print('Failed evaluation episode %d with reward %f, %d steps' % (e,reward, env.unwrapped.step_count))
            break
        
        # Move to the Next State
        currentState = nextState

# Print a Summary of the Evaluation Results
print('Completion rate %.2f with average reward %0.4f and average steps %0.2f' % (eval_counter/eval_episodes, total_reward/eval_episodes,  total_steps/eval_episodes))

writer.close()

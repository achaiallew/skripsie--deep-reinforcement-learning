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
print(device)

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

# # ---- TRAINING HYPERPARAMETERS ----
# alpha = 0.0002                   # learning rate
# episodes = 3000                  # total episodes for training (per assignment spec)
# batch_size = 128                 # neural network batch size

# # ---- Q-LEARNING HYPERPARAMETERS ----
# gamma = 0.90                     # discounting rate

# # ---- EXPLORATION HYPERPARAMETERS for Epsilon Greedy Strategy ----
# start_epsilon = 1.0              # exploration probability at start
# stop_epsilon = 0.01              # minimum exploration probability
# decay_rate = 20000               # exponential decay rate for exploration probability

# ---- MEMORY HYPERPARAMETERS ----
target_update = 2000              # no. steps bet. updating target network
mem_size = 200000                  # no. experiences the memory can keep

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

# Update Hidden Layer Size
hiddenLayerSize = (128, 128)

#============================================================================
# Epsilon-Greedy Exploration
#============================================================================
def select_action(state, explore_param, policy_net, greedy=False):
    '''Select an action using epsilon-greedy exploration.
       If greedy=True, always act greedily (used for evaluation).'''
    global steps_done

    if greedy:
        with torch.no_grad():
            print("State:")
            print(state)

            q_values = policy_net(state)
            print("Q-values:")
            print(q_values)

            action = q_values.max(1)[1].unsqueeze(0)
            print("Chosen action:", action)

            return action
   
    # Extract Parameters
    start_epsilon = explore_param[0]
    stop_epsilon = explore_param[1]
    decay_rate = explore_param[2]

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
# Optimise Model
#============================================================================
def optimise_model(train_param, writer, optimiser, policy_net, target_net):

    # Extract Parameters
    #alpha = train_param[0]
    gamma = train_param[1]
    batch_size = train_param[2]

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
    if writer is not None:
        writer.add_scalar('Loss', loss.item(), steps_done)
        writer.add_scalar('TDError', TDerrors.abs().mean().item(), steps_done)


#============================================================================
# Deep Q-Learning Function
#============================================================================
def deep_q_learning(env, train_param, explore_param, optimiser, policy_net, target_net, episodes=800, writer=None):

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
            action = select_action(state, explore_param, policy_net)
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
            if steps_done % 4 == 0:
                optimise_model(train_param, writer, optimiser, policy_net, target_net)

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

#============================================================================
# Tune Hyperparameters using Optuna
#============================================================================
def objective(trial):
    # Tune Model Hyperparameters
    alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    start_epsilon = trial.suggest_float("start_epsilon", 0.9, 1.0)
    decay_rate = trial.suggest_float("decay_rate", 10000, 25000)
    stop_epsilon = trial.suggest_float("stop_epsilon", 0.001, 0.02)

    # Group Parameters
    train_param = [alpha, gamma, batch_size]
    explore_param = [start_epsilon, stop_epsilon, decay_rate]

    # Declare Empty Rewards Array
    rewards = []
    # Start Tuning Time
    start_time = time.time()

    global steps_done

    for seed in range(2): 
        # Reset Steps Done
        steps_done = 0 

        np.random.seed(seed)
        random.seed(seed)

        # ---- RESET FRESH NETWORKS PER SEED ----
        # Instantiate the Policy Network and Target Network
        policy_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)
        target_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)
        # Copy Weights of Policy Network to Target Network
        target_net.load_state_dict(policy_net.state_dict())
        # Set Target Network to Eval Mode to not Update Parameters
        target_net.eval()
        # Adam Optimiser
        optimiser = optim.Adam(policy_net.parameters(), lr=alpha)

        # Make the Gym Environment
        env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
        env = ImgObsWrapper(env) 

        # Search the Algorithm
        mean_reward,_ = deep_q_learning(env, train_param, explore_param, optimiser, policy_net, 
                                        target_net, episodes=200, writer=None) 
        
        rewards.append(mean_reward)

        trial.report(np.mean(rewards), seed)
        if trial.should_prune():
            raise optuna.TrialPruned()
        env.close()

    
    elapsed = time.time() - start_time
    print(f'Trial Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')

    # Return the Average of the Trial Rewards
    return np.mean(rewards)


#============================================================================
# Evaluate Agent Performance
#============================================================================
def eval_model(final_policy_net):
    eval_counter = 0.0
    total_steps = 0.0
    total_reward = 0.0

    for e in range(eval_episodes):
        # Initialize the Environment and State
        current_obs, _ = env.reset()
        current_state = preprocess(current_obs)
        print(current_obs)
        print(current_state)

        # Main RL Loop
        for i in range(0, max_steps):
            # Select an Action
            action = select_action(current_state, _, final_policy_net, greedy=True)
            a = action.item()
            print("Action:", a)

            # Take Action
            obs, reward, done, truncated, info = env.step(a)
            print("Raw obs equal:", np.array_equal(current_obs, obs))
            print("Agent pos:", env.unwrapped.agent_pos)
            print("Agent dir:", env.unwrapped.agent_dir)

            # Observe a New State
            if done or truncated:
                next_state = None
            else:
                next_state = preprocess(obs)
                print("Compare Current/Next State:", torch.equal(current_state, next_state))

            # Calculate Reward
            if done or truncated:
                total_reward += reward
                total_steps += env.unwrapped.step_count
                if done:
                    print('Finished evaluation episode %d with reward %f, %d steps, reaching goal'
                        % (e, reward, env.unwrapped.step_count))
                    eval_counter += 1
                if truncated:
                    print('Failed evaluation episode %d with reward %f, %d steps'
                        % (e, reward, env.unwrapped.step_count))
                break

            # Move to the Next State
            current_state = next_state

    print('Completion rate %.2f with average reward %0.4f and average steps %0.2f'
        % (eval_counter/eval_episodes, total_reward/eval_episodes, total_steps/eval_episodes))



#============================================================================
# Run the Study
#============================================================================
if __name__ == "__main__":
        
    # Study the Model
    print("Start Hyperparameter Tuning...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials= 30)
    
    # Display the Best Parameters
    print("Best Params:", study.best_params)
    print("Best Value:", study.best_value)

    # Extract the Best Parameters
    best = study.best_params

    # Log Best Parameters
    train_param = [best["alpha"], best["gamma"], best["batch_size"]]
    decay_rate = best["decay_rate"] * (3000 / 200)  
    explore_param = [best["start_epsilon"], best["stop_epsilon"], decay_rate]

    # ---- RESET NETWORKS FOR FINAL TRAINING ----
    # Instantiate the Policy Network and Target Network
    policy_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)
    target_net = DQN(input_size, num_actions, hiddenLayerSize).to(device)
    # Copy Weights of Policy Network to Target Network
    target_net.load_state_dict(policy_net.state_dict())
    # Set Target Network to Eval Mode to not Update Parameters
    target_net.eval()
    # Adam Optimiser
    optimiser = optim.Adam(policy_net.parameters(), lr=best["alpha"])

    # Gym Environment
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped
    env = ImgObsWrapper(env)
        
    # SetUp Tensorboard
    writer = SummaryWriter()

    # Start Training
    print('Start training...')
    start_time = time.time()

    steps_done = 0

    # Train the Model
    _, final_steps = deep_q_learning(env, train_param, explore_param, optimiser, policy_net, 
                                     target_net, episodes=3000, writer=writer)

    # Done Training
    print('Done Training...')
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Training Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
    print(f'Final Steps Taken: {final_steps}')

    
    # Flush Remaining Data
    writer.flush()
    writer.close()
    print(policy_net.fc1.weight)
    print(policy_net.fc2.weight)
    print(policy_net.fc3.weight)

    # Save the Trained Model
    torch.save(policy_net, filename)
    
    # Reload the Newly Trained Model
    if exists(filename):
        final_policy_net = torch.load(filename, map_location=device, weights_only=False)
        final_policy_net.eval()
        print(f'Loaded trained model from {filename}')
    else:
        raise FileNotFoundError(f'No saved model found at {filename}.')

    # Evaluate Model Performance
    print('Starting Evaluation...')
    eval_model(final_policy_net)

    # Close the Environment
    env.close()




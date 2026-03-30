#============================================================================
# Import Modules
#============================================================================
import minigrid
import gymnasium as gym
from minigrid.wrappers import *

import random
import time
import pickle
from os.path import exists

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#============================================================================
# Define Functions
#============================================================================
# Extract Object_Idx Info Using Numpy Slicing and Reshaping
def extractObjInfo(obs):
    (rows, cols, x) = obs.shape
    temp = np.reshape(obs, [rows*cols*x, 1], 'F')[0:rows*cols]
    return np.reshape(temp, [rows, cols], 'C')

#============================================================================
# SetUp the RL Agent
#============================================================================

# Make the Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode='human').unwrapped
    # 'human' allows us to see the rendered virtual environment

# Variable for storing the Tabular Value-Function
Q = {}
filename = 'qtable.pickle'

# ---- LOAD Q-TABLE IF IT EXISTS ----
if (exists(filename)):
    print('Loading Existing Q Values...')
    # Load Data (Deserialise)
    with open(filename, 'rb') as handle:
        Q = pickle.load(handle)
        handle.close()
else:
    print('Filename %s DNE: Could Not Load Data' % filename)

# Ranges
numActions = 3 # first 3 actions
episodes = 2000
maxSteps = env.max_steps

# Wrapper - Observation will only contain Grid Information
env = ImgObsWrapper(env)

# Reset the Environment
obs, _ = env.reset()

# Extract Current State
state = extractObjInfo(obs)

'''initialise the initial values of the value-function to be zero 
    - this is a pessimistic initialisation
## note that using the numpy array of the observation will not work in practice, 
    you will need to calculate a hash-value of the current state and 
    use it as unique key into the dictionary '''

# State Hash Value
stateKey = hash(state.tobytes())
if stateKey not in Q: # prevent KeyError on Unseen States
    Q[stateKey] = np.zeros(numActions)

# Training Variables
epsilon = 0.99
epsilon_decay = 0.9999
epsilon_min = 0.01

alpha = 0.1   # learning rate
gamma = 0.99  # discount factor

# Plotting SetUp
count = 0

#============================================================================
# Main RL Loop  (Max Steps: 256)
#============================================================================
# Start Training
print('Start Training...')

# Episode Loop
for e in range(episodes):
    # Reset the Environment
    obs, _ = env.reset()

    # Extract Current State
    state = extractObjInfo(obs)

    # State Hash Value
    stateKey = hash(state.tobytes())
    if stateKey not in Q: # prevent KeyError on Unseen States
        Q[stateKey] = np.zeros(numActions)
   
    # Agent Step Loop
    for s in range(0, maxSteps):

        # Agent takes Random Action
        #//a = random.randint(0, numActions)

        #============================================================================
        # Epsilon-Greedy Exploration
        #============================================================================
        # Perform Epsilon Greedy Action
        if (random.random() < epsilon):
            # Explore Environment - Select Random Action
            a = random.randint(0, numActions-1)
        else:
            # Exploit Environment - Select Action for max of Value Function @ Current State
            a = np.argmax(Q[stateKey])


        ''' take action 'a', receive reward 'reward', and observe next state 'obs'
        'done' boolean variable that indicates if the termination state was reached
        'truncated' boolean variable indicates if episode ended before reaching termination state
        'info' information provided by the gym environment '''

        # Extract Step Information
        obs, reward, done, truncated, info = env.step(a)

        # Extract Next State from Observation
        state2 = extractObjInfo(obs)

        # Hash the Next State
        state2Key = hash(state2.tobytes())
        if state2Key not in Q:
            Q[state2Key] = np.zeros(numActions)

        #============================================================================
        # ---- Q-TABLE UPDATE (Bellman Equation) ----
        #============================================================================
        # Q-Learning
        Q[stateKey][a] = Q[stateKey][a] + alpha*(reward + gamma*np.max(Q[state2Key]) - Q[stateKey][a])

        # Decay Epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Render the Environment
        env.render()

        # Goal was Reached
        if (done == True):      
            print('Episode %d Finished Successfully:' % e)
            print('Steps: %d' % s)
            print('Reward: %f' % reward)
            break

        # No More Steps Allowed
        if (truncated == True):
            print('Episode %d Finished Unsuccessful - Truncated:' % e)
            print('Steps: %d' % s)
            print('Reward: %f' % reward)
            break

        # Move to Next State
        state = state2
        stateKey = state2Key
        count += 1

    # Write to Tensorboard
    writer.add_scalar("Reward/train", reward, count)

print('Done Training...')
# Flush Remaining Data
writer.flush()
writer.close()

# ---- SAVE Q-TABLE AFTER TRAINING ----
with open(filename, 'wb') as handle:
    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
# Import Modules
import minigrid
import gymnasium as gym
#============================================================================

# Make the Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode='human') 
    # 'human' allows us to see the rendered virtual environment
#============================================================================

# Agent takes Random Action
import random
a = random.randint(0, env.action_space.n -1)
print("Action a: %d was generated" % a)
#============================================================================


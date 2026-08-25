# Imports for Pytorh
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


# Imports for Quantization Aware Training
from torchao.quantization import quantize_, Int4WeightOnlyConfig
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig

# Imports for Writing toTensorboard
from torch.utils.tensorboard import SummaryWriter

# Imports for Environment
import gymnasium as gym
from minigrid.wrappers import *

#Imports from Basic DQN
from deep_q_network import (ReplayMemory, deep_q_learning, eval_model)

import copy

import os
from os.path import exists

import time

#============================================================================
# Settings
#============================================================================

model_trained = "dqn_trained.pth"     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- MODEL HYPERPARAMETERS ----
num_actions = 3                  # left, right, forward
input_size = 53                 # size of flattened input state (7x7 matrix of tile IDs)
mem_size = 200000

#============================================================================
# Network
#============================================================================
class DQN_Quant(nn.Module):

    def __init__(self, inputSize, numActions, hiddenLayerSize=(128,128)):
        super().__init__()

        self.fc1 = nn.Linear(inputSize, hiddenLayerSize[0])
        self.fc2 = nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1])
        self.fc3 = nn.Linear(hiddenLayerSize[1], numActions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# Update Hidden Layer Size
hiddenLayerSize = (128, 128)


#============================================================================
# Load Model
#============================================================================
if exists(model_trained):
    policy_net = DQN_Quant(input_size, num_actions, hiddenLayerSize).to(device)
    policy_net.load_state_dict(torch.load(model_trained, map_location=device))
    policy_net.cpu()
    policy_net.eval()
    print(f'Loaded trained model from {model_trained}')
else:
    raise FileNotFoundError(f'No saved model found at {model_trained}.')

# Create a Copy to Quantize
qat_policy_net = copy.deepcopy(policy_net)

# Instantiate the Target Network
qat_target_net = DQN_Quant(input_size, num_actions, hiddenLayerSize).to(device)

#============================================================================
# Prepare QAT Model
#============================================================================
# Configure Activations (INT8 Assymetric)
activation_config = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False,  is_dynamic=True)
# Configure Weights (INT4)
weight_config = IntxFakeQuantizeConfig(torch.int4, "per_channel", is_symmetric=True,  is_dynamic=True)

# Prepare Fake Quantization
qat_prepare_config = QATConfig(
    activation_config=activation_config,
    weight_config=weight_config,
    step="prepare",
)
qat_convert_config = QATConfig(
    step="convert",
)

# Fake Quatization of the Policy Network
quantize_(qat_policy_net, qat_prepare_config)
# Save the Prepared Model
torch.save(qat_policy_net.state_dict(), "dqn_qat_prepared.pth")

# Copy Weights of Policy Network to Target Network
quantize_(qat_target_net, qat_prepare_config)
qat_target_net.load_state_dict(qat_policy_net.state_dict())
# Set Target Network to Eval Mode to not Update Parameters
qat_target_net.eval()

#============================================================================
# Setup Best Parameters from Previously Trained Model
#============================================================================
## Fine-Tuning (Best Optuna Parameters for training from Scratch)
# alpha = 0.00013655199655359068
# gamma = 0.8435494470444489
# batch_size = 128

# start_epsilon = 1.0
# stop_epsilon = 0.05446496596430388
# decay_rate = 75410.74058753991 * (3000 / 500)

alpha = 0.000013655199655359068
gamma = 0.8435494470444489
batch_size = 128

start_epsilon = 0.1
stop_epsilon = 0.01
decay_rate = 10000

train_param = [alpha, gamma, batch_size]
explore_param = [start_epsilon, stop_epsilon, decay_rate]

# Memory
qat_memory = ReplayMemory(mem_size)
# Episode Size
qat_episodes = 500

# Adam Optimiser
qat_optimiser = optim.Adam(qat_policy_net.parameters(), alpha)

# Make the Gym Environment
env = gym.make('MiniGrid-Empty-8x8-v0', render_mode=None).unwrapped

# SetUp Tensorboard
writer = SummaryWriter()

# Start QAT
print('Start Quantization Aware Training (Fine-Tuning)...')
start_time = time.time()

# Train with QAT
mean_reward, success_rate, mean_steps = deep_q_learning(env, train_param, explore_param, qat_optimiser, qat_policy_net, 
                                                        qat_target_net, episodes=qat_episodes, writer=writer, memory=qat_memory)

# Done Training
print('Done Quantization Aware Training (Fine-Tuning)...')
end_time = time.time()
elapsed = end_time - start_time
print(f'Training Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
print(f'Mean Steps Taken: {mean_steps}')
print(f'Mean Reward: {mean_reward}')
print(f'Success Rate: {success_rate}')

# Flush Remaining Data
writer.flush()
writer.close()


# Save the Trained Model
torch.save(qat_policy_net.state_dict(), "dqn_qat_trained.pth")

qat_converted_net = copy.deepcopy(qat_policy_net)
qat_converted_net.eval()

# Convert the Trained Fake-Quantized Model
quantize_(qat_converted_net, qat_convert_config)

# Save the Trained Model
torch.save(qat_converted_net.state_dict(), "dqn_qat_converted.pth")

# Evaluate Model Performance
print('Starting Evaluation...')
start_time = time.time()
qat_policy_net.eval()
mean_reward, success_rate, mean_steps = eval_model(env, qat_policy_net, eval_episodes=100, print_results=True)

# Done Final Evaluation
end_time = time.time()
elapsed = end_time - start_time
print(f'Evaluation Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
print(f"Completion Rate: {success_rate:.2f}% | "
        f"Average Reward: {mean_reward:.4f} | "
        f"Average Steps: {mean_steps:.2f}"
)

# Evaluate Converted Model Performance
print('Starting Evaluation...')
start_time = time.time()
qat_converted_net.eval()
mean_reward, success_rate, mean_steps = eval_model(env, qat_converted_net, eval_episodes=100, print_results=True)

# Done Final Evaluation
end_time = time.time()
elapsed = end_time - start_time
print(f'Evaluation Done In {elapsed:.2f} s ({elapsed/60:.2f} min)')
print(f"Completion Rate: {success_rate:.2f}% | "
        f"Average Reward: {mean_reward:.4f} | "
        f"Average Steps: {mean_steps:.2f}"
)

print("Prepared layer:", type(qat_policy_net.fc1))
print("Converted layer:", type(qat_converted_net.fc1))
print("Converted weight dtype:", qat_converted_net.fc1.weight.dtype)

# Close the Environment
env.close()

# Compare File Sizes
original_size = os.path.getsize("dqn_trained.pth") / 1024**2
qat_prep_size = os.path.getsize("dqn_qat_prepared.pth") / 1024**2
qat_size = os.path.getsize("dqn_qat_trained.pth") / 1024**2
qat_quant_size = os.path.getsize("dqn_qat_quantized.pth") / 1024**2

print(
    f"Size Reduction(Prep): {original_size / qat_prep_size:.2f}x ({original_size:.2f}MB -> {qat_prep_size:.2f}MB)"
)
print(
    f"Size Reduction (Trained): {original_size / qat_size:.2f}x ({original_size:.2f}MB -> {qat_size:.2f}MB)"
)
print(
    f"Size Reduction (Qaunt Again): {original_size / qat_quant_size:.2f}x ({original_size:.2f}MB -> {qat_quant_size:.2f}MB)"
)



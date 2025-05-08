import os
import numpy as np
from typing import Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.muzero import RicochetRobotsConfig
from core.state import RicochetRobotsGame

from config import WEIGHTS_FILE_PATH


##########################
####### Helpers ##########

GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):

    def __init__(self, config: RicochetRobotsConfig):
        
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int, action_space_size: int):
        
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), 
                 g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, action_space_size))
                for (g, i) in game_pos]

    def sample_game(self) -> RicochetRobotsGame:
        
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[np.random.choice(range(len(self.buffer)))]

    def sample_position(self, game) -> int:
        
        try:
            
            # Sample position from game either uniformly or according to some priority.
            return np.random.choice(range(game.total_moves()-1))
            
        except ValueError as e:
        
            print("⚠️ Invalid Sample taken")
            print(f"Debug: {range(game.total_moves()-1)}, {game.total_moves()}")

            raise e
        
    
    def last_game(self) -> RicochetRobotsGame:
        
        return self.buffer[-1]


class NetworkOutput(NamedTuple):
    
    value: float
    reward: float
    policy_logits: Dict[int, float]
    policy_tensor: torch.Tensor
    hidden_state: torch.Tensor

        
class Network(nn.Module):    

    def __init__(self, config):
        
        super(Network, self).__init__()
        
        self.action_space_size = config.action_space_size
        hidden = config.hidden_layer_size
        self.tot_training_steps = 0

        # Representation network
        self.representation = nn.Sequential(
            nn.Linear(config.observation_space_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        # Value network
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.ReLU()
        )

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.action_space_size),
            nn.Softmax(dim=-1)
        )

        # Reward network
        self.reward = nn.Sequential(
            nn.Linear(hidden + config.action_space_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.ReLU()
        )

        # Dynamics network
        self.dynamics = nn.Sequential(
            nn.Linear(hidden + config.action_space_size, hidden),  # state + action
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        ) 
        
        print(f"Hello, I'm Leela! Ready to begin training on device {str(GPU_DEVICE).upper()} ...")
        self.to(GPU_DEVICE)
                
    def save_model(self, path=WEIGHTS_FILE_PATH):
        
        # Get absolute path to the model_weights.json file
        current_dir = os.path.dirname(__file__)
        model_weights_file = os.path.join(current_dir, "model", path)

        torch.save(self.state_dict(), model_weights_file)


    def load_model(self, path=WEIGHTS_FILE_PATH):
        
        # Get absolute path to the model_weights.json file
        current_dir = os.path.dirname(__file__)
        model_weights_file = os.path.join(current_dir, "model", path)
        
        self.load_state_dict(torch.load(model_weights_file))
        self.eval()
        
        
    def initial_inference(self, observation):
        
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32).to(GPU_DEVICE)
        
        observation = observation.unsqueeze(0).to(GPU_DEVICE)  # Add batch dimension
        hidden_state = self.representation(observation)
        
        value = self.value(hidden_state)
        policy_logits = self.policy(hidden_state)
        policy_probs = torch.softmax(policy_logits, dim=-1)

        reward = torch.tensor([[0.0]]).to(GPU_DEVICE)  # Reward is always zero on initial inference

        policy_dict = {a: policy_probs[0, a].item() for a in range(self.action_space_size)}

        
        
        return NetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_dict,
            policy_tensor=policy_probs,
            hidden_state=hidden_state
        )
    
    
    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        
        # Ensure hidden_state is a torch tensor
        if not isinstance(hidden_state, torch.Tensor):
            hidden_state = torch.tensor(hidden_state, dtype=torch.float32).to(GPU_DEVICE)
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        hidden_state = hidden_state.to(GPU_DEVICE)

        # One-hot encode the action
        action_index = int(action)  # make sure action is an integer index
        action_one_hot = F.one_hot(
            torch.tensor([action_index]).to(GPU_DEVICE), num_classes=self.action_space_size
        ).float().to(GPU_DEVICE)  # shape: [1, action_space_size]

        # Concatenate hidden state and action
        nn_input = torch.cat((hidden_state, action_one_hot), dim=1)  # shape: [1, hidden + action_space_size]

        # Dynamics network outputs next hidden state
        next_hidden_state = self.dynamics(nn_input)

        # Predict reward, value, and policy
        reward = self.reward(nn_input)
        value = self.value(next_hidden_state)
        policy = self.policy(next_hidden_state)
        policy_p = policy[0]

        # Build the policy dictionary
        policy_dict = {a: policy_p[a].item() for a in range(self.action_space_size)}

        return NetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_dict,
            policy_tensor=policy,
            hidden_state=next_hidden_state
        )
            
        
    def forward(self, observation):
        # Initial inference from raw observation
        hidden_state = self.representation(observation)
        value = self.value(hidden_state)
        policy = self.policy(hidden_state)
        
        return hidden_state, value, policy
    
    def get_weights(self):
        
        # Returns the weights of this network.

        networks = [
            self.representation,
            self.value,
            self.policy,
            self.reward,
            self.dynamics
        ]
        
        return [param for net in networks for param in net.parameters()]  
            
    def training_steps(self) -> int:
        
        # How many steps / batches the network has been trained for.
        return self.tot_training_steps


class SharedStorage:
    
    def __init__(self, config, path=WEIGHTS_FILE_PATH):
        self.networks = {}  # Dictionary to store networks by step
        self.latest_step = 0
        
        # Get absolute path to the level.json file
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, "model", path)
        
        
        self.latest_network = Network(config)
        if os.path.exists(model_path):
            self.latest_network.load_model(model_path)


    def get_latest_network(self) -> Network:
        
        return self.latest_network


    def save_network(self, step: int, network: Network):
        
        self.networks[step] = network
        if step > self.latest_step:
            self.latest_step = step
            self.latest_network = network


    def get_network(self, step: int) -> Network:
        
        return self.networks.get(step, self.latest_network)


    def update_elapsed_time(self, new_time):

        self.latest_network.save_model()
        
        # Get absolute path to the level.json file
        current_dir = os.path.dirname(__file__)
        elapsed_time_file = os.path.join(current_dir, "logs", "elapsed_time.txt")

        try:
            # Read the current elapsed time from the file (if it exists)
            with open(elapsed_time_file, 'r') as file:
                current_time = file.read().strip()
            
            # Update the time by adding the new value
            updated_time = float(current_time) + float(new_time) if current_time else new_time
            print(f"Total Elapsed Time: {updated_time} minutes [+{new_time} min/ep]\n")
            
        except FileNotFoundError:
            # If the file doesn't exist, start with the new time
            updated_time = new_time
            print(f"File not found. Starting with new time: {updated_time} minutes")
        
        # Write the updated time back to the file
        with open(elapsed_time_file, 'w') as file:
            file.write(str(updated_time))
      
      
##### End Helpers ########
##########################
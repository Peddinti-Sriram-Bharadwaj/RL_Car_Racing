# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/models/dqn.py
# ==============================================================================
# Created by: Alec Trela
# GitHub: https://github.com/artrela
# Description: Classes related to a DQN RL agent
#
# This is included as a part of the MRSD bootcamp, meant to be a primer for students
# entering their first year of the program at Carnegie Mellon University
#
# Feel free to use, modify, and share this file. Attribution is appreciated!
# For more information, visit my GitHub or
# https://github.com/RoboticsKnowledgebase/mrsd-software-bootcamp.
# ==============================================================================

from collections import namedtuple, deque
# Assuming WandBLogger is correctly defined in utils
from RL_Car_Racing.utils import WandBLogger
from typing import List, NamedTuple, Optional, Tuple
import gc
import gymnasium
import numpy as np
import random
import torch
import os
import torch.nn as nn
import torch.nn.functional as F # Added for ReLU in QNetwork forward

# ==============================================================================
# DQN Action Space Discretization
# Define your discrete action space here.
# Action (len(3)): [steering, gas, break]
# ==============================================================================
ACTION_SPACE: List[np.ndarray] = [
        np.array([    0, 0.8,    0]), # Strong Gas
        np.array([    0, 0.6,    0]), # Medium Gas
        np.array([    0, 0.4,    0]), # Light Gas
        np.array([-0.67,   0,    0]), # Strong Left
        np.array([ 0.67,   0,    0]), # Strong Right
        np.array([-0.33,   0,    0]), # Light Left
        np.array([ 0.33,   0,    0]), # Light Right
        np.array([    0,   0,  0.3]), # Strong Brake
        np.array([    0,   0, 0.15]), # Light Brake
        np.array([    0,   0,    0])  # Do Nothing
    ]

# Define the Experience namedtuple structure
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "terminal"])


class DQNAgent():

    def __init__(self, env: gymnasium.Env, experiment: dict, log: bool, pretrained_path: str = None):
        """ A DQN agent for driving the Car Racing environment.

        Args:
            env (gymnasium.Env): Car racing environment (assumed wrapped)
            experiment (dict): A set of hyper parameters used to define the agent's characteristics
            log (bool): Whether or not to log the agent's results in WandB
            pretrained_path (str, optional): Path to load pre-trained model weights. Defaults to None.
        """
        # --- Define attributes used by network initialization first ---
        self.action_space = ACTION_SPACE
        if not self.action_space:
             raise ValueError("ACTION_SPACE list cannot be empty!")
        num_actions = len(self.action_space)
        self.env = env # Store env instance

        # --- Initialize Networks ---
        # Determine input channels from env observation space (assuming wrappers are applied)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3: # Assuming (C, H, W) from FrameStack + TransformObservation
             input_channels = obs_shape[0]
        else:
             # Handle cases where observation space might not be as expected after wrappers
             print(f"Warning: Unexpected observation shape {obs_shape}. Assuming 4 input channels.")
             input_channels = 4 # Default based on FrameStack(4)

        # Policy Network (the one being trained)
        self.q_net = QNetwork(num_actions=num_actions, input_channels=input_channels)
        self.device = self.q_net.device # Get device from QNetwork instance

        # --- Load Pretrained Model if path is provided ---
        if pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"Loading pre-trained model from: {pretrained_path}")
                try:
                    state_dict = torch.load(pretrained_path, map_location=self.device)
                    self.q_net.load_state_dict(state_dict)
                    print("Pre-trained model loaded successfully into policy network (q_net).")
                except Exception as e:
                    print(f"Error loading pre-trained model: {e}. Proceeding with random weights.")
            else:
                print(f"Warning: Pretrained model path not found: {pretrained_path}. Proceeding with random weights.")
        else:
            print("No pre-trained model path provided. Starting with random weights.")
        # --- Load Pretrained Model ---

        # --- Initialize Target and Best Networks ---
        # Target Network (for stable target calculation)
        self.target_net = QNetwork(num_actions=num_actions, input_channels=input_channels).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict()) # Sync target net initially
        self.target_net.eval() # Target network is only for inference

        # Best Network (tracks the best performing policy network weights)
        self.best_net = QNetwork(num_actions=num_actions, input_channels=input_channels).to(self.device)
        self.best_net.load_state_dict(self.q_net.state_dict()) # Sync best_net initially
        self.best_net.eval()
        # --- Initialize Target and Best Networks ---


        # --- Hyperparameters and Other Attributes ---
        params = experiment['params']
        self.current_episode:int = 0
        self.total_steps: int = 0 # Start total steps at 0
        self.training_steps: int = 0 # Track training updates
        self.start_skip: int = params.get('start_skip', 0)
        self.episode_decay: int = params.get('episode_decay', 1000) # Episodes over which epsilon decays
        self.network_update: int = params.get('step_update', 4) # Env steps between training updates
        self.target_update_freq: int = params.get('target_update', 1000) # Training steps between target net updates
        self.epsilon_final: float = params.get('epsilon', 0.01) # Final epsilon value (used in decay calc)
        self.epsilon: float = 1.0 # Starting epsilon value
        # Calculate decay rate based on episode_decay (linear decay)
        if self.episode_decay <= 0:
             self.epsilon_decay_rate = 0.0 # No decay if episode_decay is non-positive
        else:
             self.epsilon_decay_rate = (self.epsilon - self.epsilon_final) / self.episode_decay

        self.gamma: float = params.get('gamma', 0.99)
        self.lr: float = params.get('learning_rate', 1e-4)
        self.batch_size:int = params.get('batch_size', 32)
        self.seed: int = params.get('random_seed', -1) # -1 might mean no fixed seed
        self.exp_replay = ExperienceReplay(params.get('mem_len', 50000))
        # self.action_space = ACTION_SPACE # Already defined above
        self.episode_actions = [0 for _ in range(num_actions)]
        # --- Hyperparameters and Other Attributes ---


        # --- Optimizer and Loss ---
        optimizer_name = params.get('optimizer', 'AdamW')
        if optimizer_name.lower() == 'adam':
            self.optim = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        elif optimizer_name.lower() == 'adamw':
             self.optim = torch.optim.AdamW(self.q_net.parameters(), lr=self.lr)
        elif optimizer_name.lower() == 'rmsprop':
             self.optim = torch.optim.RMSprop(self.q_net.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        loss_name = params.get('loss', 'MSE')
        if loss_name.lower() == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss_name.lower() == 'huber':
            self.loss_fn = torch.nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        # --- Optimizer and Loss ---


        # --- Logging ---
        self.logger = WandBLogger(experiment) if log else None
        self.max_tiles: int = 0
        # --- Logging ---

        print(f"DQNAgent initialized on device: {self.device}")
        return


    def __call__(self, s0: torch.Tensor, a0: int, r0: float, s1: torch.Tensor, t: bool)->int:
        """ Handles storing experience, triggering training, and selecting the next action.

        Args:
            s0 (torch.Tensor): The previous observation (on device).
            a0 (int): The action index taken.
            r0 (float): The reward received.
            s1 (torch.Tensor): The resulting observation (on device).
            t (bool): Was a terminal state reached (terminated or truncated).

        Returns:
            int: The index of the next action to take.
        """
        # Store experience (move tensors to CPU for storage if needed)
        s0_cpu = s0.cpu() if s0.is_cuda else s0
        s1_cpu = s1.cpu() if s1.is_cuda else s1
        self.exp_replay.storeExperience(s0_cpu, a0, r0, s1_cpu, t)

        # Increment total environment steps *after* storing experience
        self.total_steps += 1

        # Check if ready to train
        loss = None
        if len(self.exp_replay) >= self.batch_size and self.total_steps % self.network_update == 0:
            experiences = self.exp_replay.getRandomExperiences(self.batch_size)
            states, actions, rewards, next_states, terminals = self._prepareMinibatch(experiences)

            # Calculate target Q-values using the target network (Standard DQN)
            with torch.no_grad():
                 q_values_next_target = self.target_net(next_states)
                 max_q_next, _ = q_values_next_target.max(dim=1)
                 # Target is reward if terminal, otherwise reward + discounted max future Q
                 target_q_values = rewards + self.gamma * max_q_next * (~terminals) # Use boolean negation

            # Get current Q-values from policy network for the actions taken
            self.q_net.train() # Ensure policy net is in training mode
            current_q_values_all = self.q_net(states)
            # Gather Q-values corresponding to the actions taken in the batch
            current_q_values = current_q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Calculate loss
            loss_tensor = self.loss_fn(current_q_values, target_q_values)
            loss = loss_tensor.item()

            # Optimize the model
            self.optim.zero_grad()
            loss_tensor.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 1.0) # Example value
            self.optim.step()

            self.training_steps += 1

            # Update target network periodically based on training steps
            if self.training_steps % self.target_update_freq == 0:
                print(f"--- Updating target network at training step {self.training_steps} ---")
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Log step-level stats if logger exists
            if self.logger:
                self.logger.trackStatistic("losses", loss)
                self.logger.trackStatistic("q_values", current_q_values.mean().item())

            # Optional: Clean up GPU memory
            # del states, actions, rewards, next_states, terminals, target_q_values, current_q_values_all, current_q_values, loss_tensor
            # gc.collect()
            # if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Log return for the step
        if self.logger:
            self.logger.trackStatistic('returns', r0)

        # Select the next action based on the *new* state (s1)
        next_action = self.selectAction(state=s1) # Pass the next state

        # Epsilon decay (can be done per step or per episode)
        # Doing it per step here
        self._epsilonDecay()

        # Track progress (updates step counts, logs episode stats if 't' is True)
        self._trackProgress(t)

        return next_action


    def fillMemory(self, num_steps: Optional[int] = None):
        """ Fills the replay memory with experiences from random actions.

        Args:
            num_steps (Optional[int]): Number of random steps to take. Defaults to batch_size.
        """
        if num_steps is None:
            num_steps = self.batch_size
        if len(self.exp_replay) >= num_steps:
             print(f"Memory already contains {len(self.exp_replay)} experiences. Skipping fill.")
             return

        print(f"Filling replay memory with (up to) {num_steps} random steps...")
        # Use a temporary seed for filling if main seed is set, otherwise use None
        fill_seed = self.seed if self.seed != -1 else None
        state_np, info = self.env.reset(seed=fill_seed)
        # Ensure state is tensor and on device after reset
        state = torch.tensor(np.array(state_np), dtype=torch.float32).to(self.device)

        steps_added = 0
        while steps_added < num_steps:
            action_idx = random.randrange(len(self.action_space))
            action_vector = self.action_space[action_idx]
            next_state_np, reward, terminated, truncated, info = self.env.step(action_vector)
            done = terminated or truncated
            # Ensure next_state is tensor and on device
            next_state = torch.tensor(np.array(next_state_np), dtype=torch.float32).to(self.device)

            # Store experience (move to CPU for storage)
            s_cpu = state.cpu() if state.is_cuda else state
            ns_cpu = next_state.cpu() if next_state.is_cuda else next_state
            self.exp_replay.storeExperience(s_cpu, action_idx, reward, ns_cpu, done)
            steps_added += 1

            state = next_state
            if done:
                state_np, info = self.env.reset(seed=fill_seed)
                state = torch.tensor(np.array(state_np), dtype=torch.float32).to(self.device)
                if steps_added >= num_steps: # Exit if enough steps collected even after reset
                     break

        print(f"Memory filling complete. Current size: {len(self.exp_replay)}")


    def _prepareMinibatch(self, experiences: List[Experience]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Converts a list of experiences into tensors for a minibatch, moving them to the agent's device. """
        # Batch experiences
        # Ensure states are stacked correctly (should be handled by FrameStack wrapper + tensor conversion)
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=self.device) # Action indices
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        terminals = torch.tensor([exp.terminal for exp in experiences], dtype=torch.bool, device=self.device)

        return states, actions, rewards, next_states, terminals


    def yj(self, tj: bool, rj: float, sj_next: torch.Tensor) -> torch.Tensor:
        """
        Compute y_j (target Q-value) using the standard DQN update rule for a SINGLE experience.
        Note: This is often less efficient than batch computation in __call__.
        """
        if tj:
            return torch.tensor([rj], dtype=torch.float32, device=self.device)
        else:
            with torch.no_grad():
                # Ensure input tensor is on the correct device and has batch dimension
                sj_next_dev = sj_next.unsqueeze(0).to(self.device)
                q_values_next = self.target_net(sj_next_dev)
                max_q_next, _ = q_values_next.max(dim=1)
                y_j = rj + self.gamma * max_q_next
            return y_j


    def _epsilonDecay(self) -> None:
        """ Linearly anneal epsilon towards the minimum epsilon value based on episodes trained. """
        # Decay based on current episode number
        # Ensure episode_decay is positive
        if self.episode_decay > 0:
             # Linear decay from 1.0 down to self.epsilon_final over self.episode_decay episodes
             decay_fraction = max(0.0, (self.episode_decay - self.current_episode) / self.episode_decay)
             self.epsilon = self.epsilon_final + (1.0 - self.epsilon_final) * decay_fraction
        else:
             # If episode_decay is 0 or negative, just use the final epsilon
             self.epsilon = self.epsilon_final

        # Ensure epsilon doesn't go below the final value (due to float precision)
        self.epsilon = max(self.epsilon_final, self.epsilon)


    def selectAction(self, state: torch.Tensor) -> int:
        """ Select an action using epsilon-greedy strategy based on the provided state. """
        if random.random() < self.epsilon:
            # Exploration: Choose a random action index
            action_idx = random.randrange(len(self.action_space))
        else:
            # Exploitation: Choose the best action based on the policy network
            self.q_net.eval() # Set to evaluation mode for inference
            with torch.no_grad():
                # Ensure state is on the correct device (should be passed from __call__)
                q_values = self.q_net(state.unsqueeze(0)) # Add batch dimension
                action_idx = q_values.argmax(dim=1).item()
            self.q_net.train() # Set back to train mode

        # Tracking action choice (optional, can be logged if needed)
        # self.episode_actions[action_idx] += 1

        return action_idx


    def _trackProgress(self, episode_end: bool) -> None:
        """ Updates step counts and logs metrics, especially at the end of an episode. """
        # Note: total_steps is incremented in __call__

        if episode_end:
            self.current_episode += 1
            # Reset episode-specific trackers if needed (like self.episode_actions)
            # self.episode_actions = [0 for _ in range(len(self.action_space))]

            if self.logger:
                # Calculate and log episode summary statistics
                try:
                    # Check if lists are non-empty before calculating stats
                    avg_q = self.logger.averageStatistic("q_values") if self.logger.epi_stats.get("q_values") else 0.0
                    avg_ret_step = self.logger.averageStatistic("returns") if self.logger.epi_stats.get("returns") else 0.0
                    tot_ret = self.logger.sumStatistic("returns") if self.logger.epi_stats.get("returns") else 0.0
                    avg_loss = self.logger.averageStatistic("losses") if self.logger.epi_stats.get("losses") else 0.0

                    self.logger.setStatistic("epi_avg_q", avg_q)
                    # self.logger.setStatistic("epi_avg_rets_step", avg_ret_step) # Avg return per step
                    self.logger.setStatistic("epi_tot_rets", tot_ret) # Total episode return is usually more informative
                    self.logger.setStatistic("epi_avg_loss", avg_loss)

                    # Get tiles visited from the unwrapped environment
                    try:
                         tiles = self.env.unwrapped.tile_visited_count
                         self.logger.setStatistic("tiles_visited", tiles)
                    except AttributeError:
                         # print("Warning: env.unwrapped does not have tile_visited_count attribute.")
                         self.logger.setStatistic("tiles_visited", 0) # Log 0 if unavailable

                except (ZeroDivisionError, KeyError, TypeError) as e:
                    print(f"Warning: Could not calculate or log episode stats: {e}")

                # Clear lists for the next episode using the logger's method
                self.logger.trackStatistic("returns", None) # Use None to clear
                self.logger.trackStatistic("q_values", None)
                self.logger.trackStatistic("losses", None)

        # Log step-level stats (epsilon is updated per step via _epsilonDecay called in __call__)
        if self.logger:
            # self.logger.setStatistic("tot_steps", step=True) # This increments, use direct set
            self.logger.setStatistic("tot_steps", self.total_steps)
            self.logger.setStatistic("eps", self.epsilon)

        # Track max tiles visited and update best_net
        try:
            current_tiles = self.env.unwrapped.tile_visited_count
            if current_tiles > self.max_tiles:
                print(f"    New max tiles: {current_tiles} (Previous max: {self.max_tiles})")
                self.max_tiles = current_tiles
                # Update best_net weights when a new max is reached
                self.best_net.load_state_dict(self.q_net.state_dict())
                if self.logger:
                    self.logger.setStatistic("tiles_visited", self.max_tiles)
                # Optional: Save model when a new max is reached
                # self.save_model(f"./models/saved/{self.logger.run.name}_best_tiles.pth")
        except AttributeError:
            # If tile count isn't available, this check is skipped.
            pass

        return

    # Optional: Add save/load methods
    # def save_model(self, path: str):
    #     """ Saves the policy network state dictionary. """
    #     print(f"Saving model to {path}...")
    #     torch.save(self.q_net.state_dict(), path)
    #
    # def load_model(self, path: str):
    #     """ Loads the policy network state dictionary. """
    #     print(f"Loading model from {path}...")
    #     self.q_net.load_state_dict(torch.load(path, map_location=self.device))
    #     self.target_net.load_state_dict(self.q_net.state_dict()) # Sync target net
    #     self.best_net.load_state_dict(self.q_net.state_dict()) # Sync best net
    #     self.q_net.eval() # Set to eval mode after loading if using for inference
    #     self.target_net.eval()
    #     self.best_net.eval()


class QNetwork(nn.Module):
    def __init__(self, num_actions: int, input_channels: int = 4):
        """ Basic CNN architecture for Q-value approximation. """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- Calculate flattened size dynamically ---
        def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
            h = np.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
            w = np.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
            return int(h), int(w)

        input_dim = (84, 84) # Assuming 84x84 input after cropping
        h_w1 = conv_output_shape(input_dim, kernel_size=8, stride=4)
        h_w2 = conv_output_shape(h_w1, kernel_size=4, stride=2)
        h_w3 = conv_output_shape(h_w2, kernel_size=3, stride=1)
        last_conv_out_channels = 64 # Output channels of the last conv layer
        flattened_size = h_w3[0] * h_w3[1] * last_conv_out_channels

        if flattened_size <= 0:
             raise ValueError(f"Calculated flattened size is not positive ({flattened_size}). Check input dimensions and CNN parameters.")
        # --- Calculate flattened size dynamically ---

        # Define the model layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=last_conv_out_channels, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=flattened_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions)

        # Move layers to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the network. """
        # Ensure input is on the correct device
        x = x.to(self.device)
        # Normalize input
        x = x / 255.0
        # Pass through layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values


class ExperienceReplay():
    def __init__(self, memory_length: int):
        """ A memory object for an RL agent using a deque. """
        self.Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "terminal"])
        self._replay_memory = deque(maxlen=memory_length)
        self.memory_length = memory_length


    def storeExperience(self, s0: torch.Tensor, a0: int, r0: float, s1: torch.Tensor, t: bool)->None:
        """ Store an experience tuple. Tensors should ideally be on CPU. """
        # Ensure tensors are detached and on CPU before storing
        s0_cpu = s0.detach().cpu()
        s1_cpu = s1.detach().cpu()
        new_experience = self.Experience(s0_cpu, a0, r0, s1_cpu, t)
        self._replay_memory.append(new_experience)
        return


    def getRandomExperiences(self, batch_size: int)->List[Experience]:
        """ Return a list of 'batch_size' random experiences. """
        if batch_size > len(self._replay_memory):
             print(f"Warning: Requested batch size {batch_size} > memory size {len(self._replay_memory)}. Returning all memory.")
             return list(self._replay_memory)
        # Use random.sample for uniform sampling
        return random.sample(self._replay_memory, batch_size)
        # The weighted sampling below might be useful for Prioritized Experience Replay, but not standard DQN
        # return random.choices(self._replay_memory,
        #                     weights=[i+1 for i in range(len(self._replay_memory))],
        #                     k=batch_size)

    def getCurrentExperience(self) -> Optional[Experience]:
        """ Returns the most recently added experience, if any. """
        if not self._replay_memory:
            return None
        return self._replay_memory[-1]

    def getCapacity(self) -> float:
        """ Returns the percentage capacity used. """
        return len(self._replay_memory) / self.memory_length * 100

    def isFull(self) -> bool:
        """ Returns True if the memory buffer is at maximum capacity. """
        return len(self._replay_memory) == self.memory_length

    def __len__(self) -> int:
        """ Returns the current number of experiences stored. """
        return len(self._replay_memory)

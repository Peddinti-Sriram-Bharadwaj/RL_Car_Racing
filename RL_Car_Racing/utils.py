# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/utils.py
import gymnasium as gym
import os, torch, wandb, yaml
from typing import Dict, Optional
import numpy as np # Added

class WandBLogger:
    def __init__(self, experiment: dict, project_name: str="RL_Car_Racing"):
        """
        A helper class to handle WandB related functionalities.
        # ... (rest of docstring) ...
        """
        # Ensure all keys used by the agent exist here
        self.stats: Dict[str, float] = {
                    "eps": 0.,
                    "tot_steps": 0.,
                    "epi_avg_rets": 0., # Average return per step in the episode (less useful)
                    "epi_tot_rets": 0., # Total return for the episode
                    "epi_avg_q": 0.,    # Average Q value during the episode
                    "epi_avg_loss": 0., # Average loss during the episode
                    "tiles_visited": 0., # Tiles visited in the current/last training episode
                    "eval_tiles_visited": 0., # Tiles visited during the last evaluation episode
                    "eval_episode_reward": 0. # Total reward during the last evaluation episode
                    # Add "running_max_tiles" here if you want to log the overall max tiles achieved
                    # "running_max_tiles": 0.
        }
        # Tracks stats per step within an episode
        self.epi_stats: Dict[str, list] = {
                    "returns": [], # List of rewards per step
                    "q_values": [], # List of Q values per training step
                    "losses": []    # List of losses per training step
        }

        try:
            wandb.login()
            self.run = wandb.init(project=project_name,
                            name = experiment.get('name', 'default_run'), # Use .get for safety
                            config = experiment.get('params', {})) # Use .get for safety
        except Exception as e:
            print(f"Error initializing WandB: {e}. Logging disabled.")
            self.run = None # Disable logging if init fails


    def setStatistic(self, name: str, val: float=1., step: bool=False)->None:
        """ Updates a statistic in the self.stats dictionary. """
        if self.run is None: return # Don't do anything if wandb failed
        if name not in self.stats.keys():
            # Optionally add the key dynamically instead of raising error?
            # print(f"Warning: Adding new key '{name}' to stats tracker.")
            # self.stats[name] = 0.0
            # Or raise error:
            raise KeyError(f" '{name}' not in statistics tracker! Valid options are {self.stats.keys()}")

        if step:
            self.stats[name] += val
        else:
            self.stats[name] = val
        return


    def trackStatistic(self, name: str, val: Optional[float]=None)->None:
        """ Tracks per-step statistics within an episode. Clears list if val is None. """
        if self.run is None: return
        if name not in self.epi_stats.keys():
            # Optionally add the key dynamically?
            # print(f"Warning: Adding new key '{name}' to episode stats tracker.")
            # self.epi_stats[name] = []
            # Or raise error:
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.epi_stats.keys()}")

        if val is not None:
            try:
                val = float(val)
                self.epi_stats[name].append(val)
            except (ValueError, TypeError):
                 print(f"Warning: Could not convert value {val} to float for tracking statistic '{name}'.")
                 # raise NotImplementedError(f"No handling for type {type(val)} exists")
        else:
            # Clear the list when val is None (typically called at episode end)
            self.epi_stats[name].clear()
        return

    def averageStatistic(self, name: str)->float:
        """ Calculate the average of a given statistic being tracked over an episode. """
        if self.run is None: return 0.0
        if name not in self.epi_stats.keys():
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.epi_stats.keys()}")
        else:
            # Handle empty list case
            return sum(self.epi_stats[name]) / len(self.epi_stats[name]) if self.epi_stats[name] else 0.0

    def sumStatistic(self, name: str)->float:
        """ Calculate the sum of a given statistic being tracked over an episode. """
        if self.run is None: return 0.0
        if name not in self.epi_stats.keys():
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.epi_stats.keys()}")
        else:
            return sum(self.epi_stats[name])


    def sendLog(self)->None:
        """ Send the current self.stats dictionary to the wandb server. """
        if self.run: # Only log if wandb was initialized successfully
            wandb.log(self.stats)


def parse_config(path: str)->dict:
    """ Given a path to a yaml file, return a dictionary object """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration not present at {path}!")
    else:
        with open(path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
            # config_file.close() # 'with open' handles closing automatically
        return config_dict


def wrap_env(env: gym.Env, experiment_name: str, split: str="train", record_t: int=1)->gym.Env:
    """ Wrap an environment with given wrappers from the Gymnasium library. """

    # Apply observation wrappers first
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    # Handle LazyFrames object from FrameStack before converting to tensor
    env = gym.wrappers.TransformObservation(env, lambda x: torch.tensor(np.array(x), dtype=torch.float32), env.observation_space)
    # Apply cropping *after* converting to tensor
    env = gym.wrappers.TransformObservation(env, lambda x: x[:, :84, :84], env.observation_space)


    # Conditionally apply RecordVideo wrapper
    if record_t > 0 and env.render_mode != 'human':
        def custom_episode_trigger(episode_id):
            return episode_id % record_t == 0
        env = gym.wrappers.RecordVideo(env,
                                       video_folder=f"./videos/{experiment_name}/{split}/",
                                       episode_trigger=custom_episode_trigger,
                                       disable_logger=True)
    return env

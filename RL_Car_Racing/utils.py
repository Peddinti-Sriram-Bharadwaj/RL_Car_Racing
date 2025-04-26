# ==============================================================================
# Created by: Alec Trela
# GitHub: https://github.com/artrela
# Description: A set of utilies functions that may be helpful for training an RL 
# agent.
#
# This is included as a part of the MRSD bootcamp, meant to be a primer for students
# entering their first year of the program at Carnegie Mellon University 
# 
# Feel free to use, modify, and share this file. Attribution is appreciated! 
# For more information, visit my GitHub or 
# https://github.com/RoboticsKnowledgebase/mrsd-software-bootcamp.
# ==============================================================================

import gymnasium as gym
import os, torch, wandb, yaml
from typing import Dict, Optional
import numpy as np


class WandBLogger:
    def __init__(self, experiment: dict, project_name: str="RL_Car_Racing"):
        """
        A helper class to handle WandB related functionalities.
        # ... (rest of docstring) ...
        """
        self.stats: Dict[str, float] = {
                    "eps": 0.,
                    "tot_steps": 0.,
                    "epi_avg_rets": 0., # Average return per step in the episode
                    "epi_tot_rets": 0., # Total return for the episode
                    "epi_avg_q": 0.,    # Average Q value during the episode
                    "epi_avg_loss": 0., # Average loss during the episode
                    "tiles_visited": 0., # Max tiles visited during training
                    "eval_tiles_visited": 0., # Tiles visited during evaluation
                    "eval_episode_reward": 0. # <-- ADD THIS KEY
        }
        self.epi_stats: Dict[str, list] = {
                    "returns": [], # List of rewards per step
                    "q_values": [], # List of Q values per training step
                    "losses": []    # List of losses per training step
        }

        wandb.login()
        self.run = wandb.init(project=project_name,
                        name = experiment['name'],
                        config = experiment['params'])
        
        
    def setStatistic(self, name: str, val: float=1., step: bool=False)->None:
        """ For the statistics given in the __init__ function, which will be send to 
        WandB logs at a self.sendLog() call, update the value. 

        Args:
            name (str): The key to update in the statistics tracker
            val (float, optional): What to set the statistic to. Defaults to 1.
            step (bool, optional): Should I step the statistic or override it? 
            A val of 1 and a step==True will increment self.stats[name] += 1. Defaults to True.

        Raises:
            KeyError: If the statistic is not existing, then throw and error showing which
            stats are tracked
        """
        if name not in self.stats.keys():
            raise KeyError(f" '{name}' not in statistics tracker! Valid options are {self.stats.keys()}")
        else:
            if step:
                self.stats[name] += val
            else:
                self.stats[name] = val
        return
    
    
    def trackStatistic(self, name: str, val: Optional[float]=None)->None:
        """Meant to track statistics that occur over an episode run. 

        Args:
            name (str): The key to update in the *episode* statistics tracker 
            val (float | list, optional): If a float value given, append to the list for 
            tracking episode statistics. If a list is given, override the dictionary value. 
            Provide no value to reset the list at the given key. Defaults to [].

        Raises:
            KeyError: If the statistic is not existing, then throw and error showing which
            stats are tracked
        """
        if name not in self.epi_stats.keys():
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.epi_stats.keys()}")
        else:
            if val is not None: 
                try: 
                    val = float(val)
                    self.epi_stats[name].append(val)
                except:
                    raise NotImplementedError(f"No handling for type {type(val)} exists")
            else:
                self.epi_stats[name].clear()
            
        return
    
    def averageStatistic(self, name: str)->float:
        """Calculate the average of a given statistic being tracked over an episode. 

        Args:
            name (str): The key to obtain the average for in the *episode* statistics tracker

        Raises:
            KeyError:  If the statistic is not existing, then throw and error showing which
            stats are tracked

        Returns:
            float: return the average of the statistic
        """
        if name not in self.epi_stats.keys():
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.epi_stats.keys()}")
        else:
            return sum(self.epi_stats[name]) / len(self.epi_stats[name])
        
    def sumStatistic(self, name: str)->float:
        """Calculate the sum of a given statistic being tracked over an episode. 

        Args:
            name (str): The key to obtain the average for in the *episode* statistics tracker

        Raises:
            KeyError:  If the statistic is not existing, then throw and error showing which
            stats are tracked

        Returns:
            float: return the sum of the statistic
        """
        if name not in self.epi_stats.keys():
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.epi_stats.keys()}")
        else:
            return sum(self.epi_stats[name])
            
            
    def sendLog(self)->None:
        """
        Send the log to the wandb server
        """
        wandb.log(self.stats)
        

def parse_config(path: str)->dict:
    """Given a path to a yaml file, return a dictionary object

    Args:
        path (str): A proposed path to a configuration path

    Raises:
        FileNotFoundError: If the file is not found, prints the path given. 

    Returns:
        dict: A parsed configuration yaml file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration not present at {path}!")
    else:
        with open(path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
            config_file.close()
    
        return config_dict
    
    
# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/utils.py

# ... (other imports and classes) ...

def wrap_env(env: gym.Env, experiment_name: str, split: str = "train", record_t: int = 1) -> gym.Env:
    """
    Wrap an environment with grayscale, stacking, tensor conversion, cropping,
    and optional video recording.

    Args:
        env (gym.Env): The Gymnasium environment.
        experiment_name (str): Experiment name for video saving.
        split (str): "train" or "test" split.
        record_t (int): Frequency of recording episodes (0 disables recording).

    Returns:
        gym.Env: Wrapped environment.
    """

    # Convert to grayscale
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)

    # Stack frames
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    # Convert LazyFrames to torch tensor
    env = gym.wrappers.TransformObservation(
        env,
        lambda x: torch.tensor(np.array(x), dtype=torch.float32),
        observation_space=None  # Let it auto-infer
    )

    # Crop the tensor
    env = gym.wrappers.TransformObservation(
        env,
        lambda x: x[:, :84, :84],
        observation_space=None
    )

    # Record video if needed
    if record_t > 0 and getattr(env, "render_mode", None) != 'human':
        def custom_episode_trigger(episode_id):
            return episode_id % record_t == 0

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=f"./videos/{experiment_name}/{split}/",
            episode_trigger=custom_episode_trigger,
            disable_logger=True
        )

    return env






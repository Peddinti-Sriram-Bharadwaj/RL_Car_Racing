# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/main.py
import argparse
import gymnasium as gym
import torch # Added
import os # Added

# Assuming utils is in the same directory or accessible via PYTHONPATH
from . import utils
# Adjust imports if needed based on your project structure
from RL_Car_Racing.models.dqn import DQNAgent
from RL_Car_Racing.models.ddqn import DDQNAgent
from RL_Car_Racing.models.dqn import ACTION_SPACE
from typing import Union, Tuple # Added for type hinting

def main(experiment: dict, debug: bool, pretrained_path: str = None)->None: # Added pretrained_path
    """
    Train an RL agent to drive in the Gymnasium environment 'CarRacing-v3', using hyperparameters specified in a
    configuration yaml file. See 'RL_Car_Racing/config/default.yaml' as a starting point.

    Args:
        experiment (dict): A set of parameters to define the experiement. High level markers are 'params' and
        'name'.
        debug (bool): Forgoes wandb logging for debugging purposes
        pretrained_path (str, optional): Path to load pre-trained model weights. Defaults to None.
    """
    params = experiment['params'] # Get params dict

    # --- Environment Setup (Use continuous=True as agent expects it) ---
    # Note: The environment needs continuous=True because your ACTION_SPACE
    # contains continuous vectors that the agent passes to env.step()
    # Using CarRacing-v3 as specified in collect_data.py and train_imitation.py
    train_env_raw = gym.make("CarRacing-v3", render_mode='rgb_array', domain_randomize=False, continuous=True)
    train_env = utils.wrap_env(train_env_raw, experiment['name'], record_t=experiment.get('record_video', -1)) # Use .get for safety

    test_env_raw = gym.make("CarRacing-v3", render_mode='rgb_array', domain_randomize=False, continuous=True)
    test_env = utils.wrap_env(test_env_raw, experiment['name'], split="eval", record_t=experiment.get('record_video', -1))
    # --- Environment Setup ---

    # --- Agent Initialization ---
    # Separate positional and keyword arguments correctly
    agent_pos_args = (train_env, experiment) # Positional arguments only
    agent_kwargs = {
        'log': not debug,                # Add 'log' here
        'pretrained_path': pretrained_path
    }

    if params['model'] == 'DQN':
        # Pass positional args with * and keyword args with **
        agent = DQNAgent(*agent_pos_args, **agent_kwargs)
    elif params['model'] == 'DDQN':
        # Ensure DDQNAgent also accepts pretrained_path and log in its __init__
        # You might need to update DDQNAgent.__init__ signature
        agent = DDQNAgent(*agent_pos_args, **agent_kwargs)
    else:
        raise NotImplementedError(f"Model type {params['model']} not implemented.")
    # --- Agent Initialization ---

    # --- Fill Memory (Optional but recommended) ---
    # Fill memory if not loading a pretrained model and if replay buffer isn't full enough
    if not pretrained_path and len(agent.exp_replay) < agent.batch_size:
         print("Filling agent memory before training...")
         # Make sure fillMemory is implemented correctly in the agent class
         try:
            agent.fillMemory()
            print(f"Memory filling complete. Current size: {len(agent.exp_replay)}")
         except AttributeError:
            print("Warning: Agent does not have a fillMemory() method. Skipping memory fill.")
         except Exception as e:
            print(f"Error during memory fill: {e}")

    # --- Training Loop ---
    num_episodes = params.get('num_episodes', 1000) # Default if not in config
    eval_interval = params.get('eval_interval', 50) # Default eval frequency

    for e in range(num_episodes):
        episode_num = e + 1
        start_str = 15*"=" + f" Episode {episode_num}/{num_episodes} Start " + 15*"="
        print(start_str)

        # Train for one episode
        # Pass necessary params from config if train function needs them
        train(agent, train_env,
              start_skip=params.get('start_skip', 0), # Default if not in config
              stacked_neg=params.get('stacked_neg', -1)) # Default if not in config

        # Evaluate periodically
        if episode_num % eval_interval == 0:
             eval(agent, test_env,
                  start_skip=params.get('start_skip', 0)) # Default if not in config

        # Log episode metrics via agent's logger
        if agent.logger:
            agent.logger.sendLog() # Agent's _trackProgress should prepare stats

        print("=" * len(start_str), "\n")
    # --- Training Loop ---

    print("Training finished.")
    train_env.close()
    test_env.close()
    if agent.logger:
         # Ensure wandb run finishes properly
         try:
            agent.logger.run.finish()
         except AttributeError:
            print("WandB run could not be finished (logger or run object missing).")


# --- train() and eval() functions ---
# These functions now just run the episode loop, agent handles internal logic

def train(agent: Union[DQNAgent, DDQNAgent], env: gym.Env, start_skip: int, stacked_neg: int)->None:
    """ Train the agent for one episode. """
    # Seed is now handled within the agent or main setup if needed globally
    prev_observation, info = env.reset() # Returns tensor from wrapper
    terminated = truncated = False

    # Action index starts invalid, agent selects first valid one
    action_idx = -1 # Use index now
    consec_neg, step = 0, 0
    total_reward = 0.0 # Track reward for info

    while not (terminated or truncated):
        step += 1

        # If first step, agent needs to select initial action based on initial state
        if action_idx == -1:
             # Pass state to selectAction for exploitation/exploration based on it
             action_idx = agent.selectAction(state=prev_observation)

        # Get the continuous action vector for the environment step
        # Ensure agent.action_space is correctly populated
        try:
            action_vector = agent.action_space[action_idx]
        except IndexError:
            print(f"Error: Invalid action index {action_idx} selected. Max index: {len(agent.action_space)-1}")
            # Handle error, maybe default to a safe action or re-select
            action_idx = agent.selectAction(state=prev_observation) # Try selecting again
            action_vector = agent.action_space[action_idx]


        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action_vector)
        total_reward += reward

        # Skip initial frames for storing/training, but still step env
        if step < start_skip:
            # Need to select next action even if skipping training step
            # Pass the new observation to selectAction
            action_idx = agent.selectAction(state=observation)
            prev_observation = observation.detach().clone() # Keep track of state
            continue

        # Process reward and negative stacking for early termination check
        # Note: The actual reward passed to the agent might be the raw reward
        if reward < 0:
            consec_neg += 1
        else:
            consec_neg = 0

        # Check for early termination due to stacked negative rewards
        if stacked_neg > 0 and consec_neg >= stacked_neg: # Use >= for clarity
            print(f"    Terminating early due to {consec_neg} consecutive negative rewards.")
            truncated = True # Use truncated for early stops not part of MDP terminal state
            info["Reason"] = "Stacked Negative Rewards"

        # Agent handles storing experience, training, and selecting NEXT action
        # Pass the raw reward 'reward' to the agent's learning mechanism
        # The agent's __call__ method should return the *next* action index
        next_action_idx = agent(prev_observation, action_idx, reward, observation, terminated or truncated)

        # Update for next iteration
        prev_observation = observation.detach().clone() # Clone to prevent modification issues
        action_idx = next_action_idx # Use the action selected by the agent for the next step

        # Check termination conditions after agent call
        if terminated or truncated:
            print(f"[TRAIN] | Episode End | Steps: {step}, Skipped: {start_skip}, Trained Steps: {max(0, step - start_skip)}, Total Reward: {total_reward:.2f}")
            # print(f"[TRAIN] | Info: {info}") # Info can be verbose
            try:
                print(f"[TRAIN] | Tiles Visited: {env.unwrapped.tile_visited_count}")
            except AttributeError:
                print("[TRAIN] | Tiles Visited: N/A")
            # Agent's _trackProgress(episode_end=True) should be called within agent.__call__ or similar
            return


def eval(agent: Union[DQNAgent, DDQNAgent], env: gym.Env, start_skip: int)->None:
    """ Evaluate the agent using the best network found so far (greedy policy). """
    print("--- Starting Evaluation ---")
    prev_observation, info = env.reset()
    terminated = truncated = False

    action_idx = -1
    step = 0
    total_reward = 0.0

    # Ensure agent uses greedy policy during evaluation
    original_epsilon = agent.epsilon # Store original epsilon
    agent.epsilon = 0.0 # Set epsilon to 0 for greedy actions

    try: # Use try...finally to restore epsilon
        while not (terminated or truncated):
            step += 1

            # Select action greedily using the 'best_net' or policy net in eval mode
            # The selectAction method should handle the case when state is provided (greedy)
            action_idx = agent.selectAction(state=prev_observation)

            # Get the continuous action vector
            try:
                action_vector = agent.action_space[action_idx]
            except IndexError:
                print(f"[EVAL] Error: Invalid action index {action_idx}. Max index: {len(agent.action_space)-1}")
                break # Stop evaluation if agent selects invalid action

            observation, reward, terminated, truncated, info = env.step(action_vector)
            total_reward += reward

            # Skip initial frames (no training happens here anyway)
            if step < start_skip:
                prev_observation = observation.detach().clone()
                continue

            prev_observation = observation.detach().clone()

            # Optional: Render during evaluation
            # env.render()

            if terminated or truncated:
                print(f"[EVAL] | Episode End | Steps: {step}, Skipped: {start_skip}, Total Reward: {total_reward:.2f}")
                # print(f"[EVAL] | Info: {info}")
                try:
                    tiles_visited = env.unwrapped.tile_visited_count
                    print(f"[EVAL] | Tiles Visited: {tiles_visited}")
                    # Log final eval metrics if logger exists
                    if agent.logger:
                         agent.logger.setStatistic('eval_episode_reward', total_reward)
                         agent.logger.setStatistic('eval_tiles_visited', tiles_visited)
                except AttributeError:
                    print("[EVAL] | Tiles Visited: N/A")
                    if agent.logger:
                         agent.logger.setStatistic('eval_episode_reward', total_reward)

                print("--- Evaluation Finished ---")
                return # Exit eval function

    finally:
        # Restore original epsilon after evaluation finishes or errors out
        agent.epsilon = original_epsilon
        print(f"--- Restored agent epsilon to: {agent.epsilon:.4f} ---")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running training cycle for RL agent on Car Racing Gymnasium Environment")

    # Use relative path assuming execution from project root
    parser.add_argument("--config", "-c", default="RL_Car_Racing/config/default.yaml", type=str,
                    help="Path to yaml file used to establish an experiment.")
    parser.add_argument("--debug", "-d",
                        help="Removes wandb logging for debugging purposes.", action="store_true")
    # --- Add argument for pretrained model ---
    parser.add_argument("--load-pretrained", type=str, default=None,
                        help="Path to the imitation learning pre-trained model (.pth file).")
    # --- Add argument for pretrained model ---

    args = parser.parse_args()

    # Load configuration using the utility function
    try:
        experiment: dict = utils.parse_config(args.config)
    except FileNotFoundError as e:
        print(e)
        exit(1) # Exit if config file not found
    except Exception as e:
        print(f"Error parsing config file {args.config}: {e}")
        exit(1)

    # Pass the pretrained path to main
    main(experiment, args.debug, pretrained_path=args.load_pretrained)

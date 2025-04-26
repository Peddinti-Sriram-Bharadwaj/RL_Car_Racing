# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/main.py
import argparse
import gymnasium as gym
import torch
import os
import sys # Added for sys.exit

# Assuming utils is in the same directory or accessible via PYTHONPATH
import utils
# Adjust imports if needed based on your project structure
from RL_Car_Racing.models.dqn import DQNAgent
from RL_Car_Racing.models.ddqn import DDQNAgent # Ensure DDQN is implemented similarly
from typing import Union, Tuple

# Default path for the imitation model
DEFAULT_IMITATION_MODEL_PATH = "imitation_model.pth"

def main(experiment: dict, debug: bool, pretrained_path: str = None, use_cost_penalty: bool = False)->None: # Added use_cost_penalty
    """
    Train an RL agent ...

    Args:
        experiment (dict): ...
        debug (bool): ...
        pretrained_path (str, optional): Path to load pre-trained model weights. Defaults to None.
        use_cost_penalty (bool): Whether to enable the cost penalty mechanism. Defaults to False.
    """
    params = experiment['params'] # Get params dict

    # --- Override config based on menu choice ---
    # The agent will read these from the modified experiment dict
    params['use_cost_penalty'] = use_cost_penalty
    if use_cost_penalty:
         print("INFO: Cost penalty enabled via menu choice.")
    # --- Override config based on menu choice ---


    # --- Environment Setup ---
    train_env_raw = gym.make("CarRacing-v3", render_mode='rgb_array', domain_randomize=False, continuous=True)
    train_env = utils.wrap_env(train_env_raw, experiment['name'], record_t=experiment.get('record_video', -1))

    test_env_raw = gym.make("CarRacing-v3", render_mode='rgb_array', domain_randomize=False, continuous=True)
    test_env = utils.wrap_env(test_env_raw, experiment['name'], split="eval", record_t=experiment.get('record_video', -1))
    # --- Environment Setup ---

    # --- Agent Initialization ---
    agent_pos_args = (train_env, experiment) # Pass the potentially modified experiment dict
    agent_kwargs = {
        'log': not debug,
        'pretrained_path': pretrained_path
    }

    # Determine agent type from config (menu doesn't change this directly)
    agent_model_type = params.get('model', 'DQN').upper()
    print(f"INFO: Initializing agent model type: {agent_model_type}")

    if agent_model_type == 'DQN':
        agent = DQNAgent(*agent_pos_args, **agent_kwargs)
    elif agent_model_type == 'DDQN':
        # Ensure DDQNAgent also handles 'use_cost_penalty' if needed,
        # or inherits the __call__ logic correctly.
        agent = DDQNAgent(*agent_pos_args, **agent_kwargs)
    else:
        raise NotImplementedError(f"Model type {agent_model_type} not implemented.")
    # --- Agent Initialization ---

    # --- Fill Memory ---
    if not pretrained_path and hasattr(agent, 'exp_replay') and hasattr(agent, 'batch_size') and len(agent.exp_replay) < agent.batch_size:
         print("Filling agent memory before training...")
         try:
            agent.fillMemory()
            print(f"Memory filling complete. Current size: {len(agent.exp_replay)}")
         except AttributeError:
            print("Warning: Agent does not have a fillMemory() method. Skipping memory fill.")
         except Exception as e:
            print(f"Error during memory fill: {e}")

    # --- Training Loop ---
    num_episodes = params.get('num_episodes', 1000)
    eval_interval = params.get('eval_interval', 50)

    for e in range(num_episodes):
        episode_num = e + 1
        start_str = 15*"=" + f" Episode {episode_num}/{num_episodes} Start " + 15*"="
        print(start_str)

        # Train for one episode
        train(agent, train_env,
              start_skip=params.get('start_skip', 0),
              stacked_neg=params.get('stacked_neg', -1))

        # Evaluate periodically
        if (e + 1) % eval_interval == 0:
             eval(agent, test_env,
                  start_skip=params.get('start_skip', 0))

        # Log episode metrics
        if agent.logger:
            agent.logger.sendLog()

        print("=" * len(start_str), "\n")
    # --- Training Loop ---

    print("Training finished.")
    train_env.close()
    test_env.close()
    if agent.logger:
         try:
            agent.logger.run.finish()
         except AttributeError:
            print("WandB run could not be finished (logger or run object missing).")


# --- train() and eval() functions ---
def train(agent: Union[DQNAgent, DDQNAgent], env: gym.Env, start_skip: int, stacked_neg: int)->None:
    """ Train the agent for one episode. """
    prev_observation, info = env.reset()
    terminated = truncated = False
    action_idx = -1
    consec_neg, step = 0, 0
    total_reward = 0.0

    while not (terminated or truncated):
        step += 1
        if action_idx == -1:
             action_idx = agent.selectAction(state=prev_observation)
        try:
            action_vector = agent.action_space[action_idx]
        except IndexError:
            print(f"Error: Invalid action index {action_idx} selected. Max index: {len(agent.action_space)-1}")
            action_idx = agent.selectAction(state=prev_observation)
            action_vector = agent.action_space[action_idx]

        observation, reward, terminated, truncated, info = env.step(action_vector)
        total_reward += reward

        if step < start_skip:
            action_idx = agent.selectAction(state=observation)
            prev_observation = observation.detach().clone()
            continue

        if reward < 0:
            consec_neg += 1
        else:
            consec_neg = 0

        if stacked_neg > 0 and consec_neg >= stacked_neg:
            print(f"    Terminating early due to {consec_neg} consecutive negative rewards.")
            truncated = True
            info["Reason"] = "Stacked Negative Rewards"

        # Agent __call__ handles augmented reward logic internally if enabled
        next_action_idx = agent(prev_observation, action_idx, reward, observation, terminated or truncated)
        prev_observation = observation.detach().clone()
        action_idx = next_action_idx

        if terminated or truncated:
            print(f"[TRAIN] | Episode End | Steps: {step}, Skipped: {start_skip}, Trained Steps: {max(0, step - start_skip)}, Total Reward: {total_reward:.2f}")
            try:
                print(f"[TRAIN] | Tiles Visited: {env.unwrapped.tile_visited_count}")
            except AttributeError:
                print("[TRAIN] | Tiles Visited: N/A")
            return


def eval(agent: Union[DQNAgent, DDQNAgent], env: gym.Env, start_skip: int)->None:
    """ Evaluate the agent using the best network found so far (greedy policy). """
    print("--- Starting Evaluation ---")
    prev_observation, info = env.reset()
    terminated = truncated = False
    action_idx = -1
    step = 0
    total_reward = 0.0
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        while not (terminated or truncated):
            step += 1
            action_idx = agent.selectAction(state=prev_observation)
            try:
                action_vector = agent.action_space[action_idx]
            except IndexError:
                print(f"[EVAL] Error: Invalid action index {action_idx}. Max index: {len(agent.action_space)-1}")
                break

            observation, reward, terminated, truncated, info = env.step(action_vector)
            total_reward += reward

            if step < start_skip:
                prev_observation = observation.detach().clone()
                continue

            prev_observation = observation.detach().clone()

            if terminated or truncated:
                print(f"[EVAL] | Episode End | Steps: {step}, Skipped: {start_skip}, Total Reward: {total_reward:.2f}")
                try:
                    tiles_visited = env.unwrapped.tile_visited_count
                    print(f"[EVAL] | Tiles Visited: {tiles_visited}")
                    if agent.logger:
                         agent.logger.setStatistic('eval_episode_reward', total_reward)
                         agent.logger.setStatistic('eval_tiles_visited', tiles_visited)
                except AttributeError:
                    print("[EVAL] | Tiles Visited: N/A")
                    if agent.logger:
                         agent.logger.setStatistic('eval_episode_reward', total_reward)
                print("--- Evaluation Finished ---")
                return
    finally:
        agent.epsilon = original_epsilon
        print(f"--- Restored agent epsilon to: {agent.epsilon:.4f} ---")
# --- train() and eval() functions ---


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running training cycle for RL agent on Car Racing Gymnasium Environment")

    parser.add_argument("--config", "-c", default="RL_Car_Racing/config/default.yaml", type=str,
                    help="Path to yaml file used to establish an experiment.")
    parser.add_argument("--debug", "-d",
                        help="Removes wandb logging for debugging purposes.", action="store_true")

    args = parser.parse_args()

    # --- Load Base Configuration ---
    try:
        experiment: dict = utils.parse_config(args.config)
        if 'params' not in experiment: experiment['params'] = {} # Ensure params dict exists
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing config file {args.config}: {e}")
        sys.exit(1)
    # --- Load Base Configuration ---


    # --- Interactive Menu ---
    pretrained_path = None
    use_cost_penalty = False
    agent_model_type = experiment['params'].get('model', 'DQN').upper() # Get model from config

    while True:
        print("\nChoose training mode:")
        print(f"  1: Train {agent_model_type} from scratch (Standard Reward)")
        print(f"  2: Train {agent_model_type} with Imitation Start (Standard Reward)")
        print(f"  3: Train {agent_model_type} with Imitation Start + Cost Penalty")

        choice = input("Enter choice (1, 2, or 3): ").strip()

        if choice == '1':
            print(f"Selected: Train {agent_model_type} from scratch.")
            pretrained_path = None
            use_cost_penalty = False
            break
        elif choice == '2' or choice == '3':
            imitation_model_path = DEFAULT_IMITATION_MODEL_PATH
            if os.path.exists(imitation_model_path):
                pretrained_path = imitation_model_path
                if choice == '2':
                    print(f"Selected: Train {agent_model_type} with Imitation Start ({pretrained_path}).")
                    use_cost_penalty = False
                else: # Choice == '3'
                    print(f"Selected: Train {agent_model_type} with Imitation Start ({pretrained_path}) + Cost Penalty.")
                    use_cost_penalty = True
                break
            else:
                print(f"Error: Imitation model '{imitation_model_path}' not found for options 2 or 3.")
                print("Please ensure train_imitation.py has been run successfully.")
                retry = input("Go back to menu? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Exiting.")
                    sys.exit(1)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    # --- Interactive Menu ---


    # Pass the determined settings to main
    main(experiment, args.debug, pretrained_path=pretrained_path, use_cost_penalty=use_cost_penalty)

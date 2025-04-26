# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/run_agent.py
import gymnasium as gym
import torch
import argparse
import time
import numpy as np
import os

# --- Imports from your project ---
# Assuming utils is in the same directory or accessible via PYTHONPATH
from RL_Car_Racing import utils
# Adjust imports if needed based on your project structure
from RL_Car_Racing.models.dqn import DQNAgent, QNetwork, ACTION_SPACE
# from RL_Car_Racing.models.ddqn import DDQNAgent # Uncomment if you want to load DDQN
# --- Imports from your project ---


def run_demonstration(model_path: str, num_episodes: int = 1, agent_type: str = 'DQN'):
    """Loads a trained agent and runs it in the environment with human rendering."""

    print(f"--- Running Demonstration ---")
    print(f"Loading model: {model_path}")
    print(f"Agent type: {agent_type}")
    print(f"Episodes: {num_episodes}")
    print("-----------------------------")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # --- Environment Setup ---
    # Use render_mode='human' to see the window
    # Use continuous=True as the agent/action space expects it
    env_raw = gym.make("CarRacing-v3", render_mode='human', continuous=True)

    # Apply the same wrappers used during training/data collection
    # Pass dummy experiment details, disable recording
    dummy_experiment = {'name': 'demonstration', 'params': {}} # Need params for logger placeholder if agent uses it
    env = utils.wrap_env(env_raw, dummy_experiment['name'], record_t=-1)
    # --- Environment Setup ---

    # --- Agent Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a dummy experiment dict just for agent init (config params not needed for eval)
    dummy_config = {'params': {}} # Agent might expect params dict

    # Instantiate the correct agent type
    if agent_type.upper() == 'DQN':
         # Pass dummy config, disable logging, no pretrained path needed here
        agent = DQNAgent(env, dummy_config, log=False)
    # elif agent_type.upper() == 'DDQN':
    #      agent = DDQNAgent(env, dummy_config, log=False) # Ensure DDQNAgent exists and init matches
    else:
        print(f"Error: Unknown agent type '{agent_type}'")
        env.close()
        return

    # Load the saved state dictionary
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Load into the policy network (q_net)
        agent.q_net.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        env.close()
        return

    # Set agent to evaluation mode (greedy actions)
    agent.q_net.eval()
    agent.epsilon = 0.0 # Ensure greedy policy
    # --- Agent Initialization ---


    # --- Run Episodes ---
    try:
        for episode in range(num_episodes):
            print(f"\nStarting Episode {episode + 1}/{num_episodes}")
            # Reset returns tensor state from wrappers
            state, info = env.reset()
            terminated = truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                # Select action greedily
                action_idx = agent.selectAction(state=state)

                # Get the corresponding action vector
                action_vector = agent.action_space[action_idx]

                # Step the environment
                next_state, reward, terminated, truncated, info = env.step(action_vector)
                total_reward += reward
                steps += 1

                # Update state
                state = next_state

                # Small delay to make rendering smoother/slower
                time.sleep(0.02)

            print(f"Episode {episode + 1} finished.")
            print(f"  Steps: {steps}")
            print(f"  Total Reward: {total_reward:.2f}")
            try:
                print(f"  Tiles Visited: {env.unwrapped.tile_visited_count}")
            except AttributeError:
                pass # Ignore if tiles not available

    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    finally:
        print("Closing environment.")
        env.close()
    # --- Run Episodes ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained Car Racing agent with rendering.")
    parser.add_argument("--model-path", "-m", type=str, required=True,
                        help="Path to the saved model weights (.pth file). E.g., imitation_model.pth")
    parser.add_argument("--episodes", "-e", type=int, default=3,
                        help="Number of episodes to run.")
    # parser.add_argument("--agent-type", "-a", type=str, default="DQN", choices=["DQN", "DDQN"],
    #                     help="Type of agent the model belongs to.") # Add if supporting DDQN

    args = parser.parse_args()

    run_demonstration(model_path=args.model_path,
                      num_episodes=args.episodes)
                      # agent_type=args.agent_type) # Add if supporting DDQN

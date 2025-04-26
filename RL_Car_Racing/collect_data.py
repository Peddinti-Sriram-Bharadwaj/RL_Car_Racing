# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/collect_data.py
import gymnasium as gym
import numpy as np
from pynput import keyboard # Using pynput
import time
import pickle
import os
import torch
import threading
import sys # Added for sys.exit

# --- Important: Make sure these match your dqn.py and utils.py ---
from RL_Car_Racing.models.dqn import ACTION_SPACE
from RL_Car_Racing.utils import wrap_env
# --- Important: Make sure these match your dqn.py and utils.py ---

# --- Configuration ---
OUTPUT_FILENAME = "human_demonstrations.pkl"
ENV_ID = "CarRacing-v3" # Keep consistent
# --- Configuration ---

# --- Key Mapping (Using pynput Key objects and characters) ---
key_to_action_idx = {
    keyboard.Key.up: 0,      # Strong Gas
    'w': 1,                  # Medium Gas
    's': 2,                  # Light Gas
    keyboard.Key.left: 3,    # Strong Left
    'a': 5,                  # Light Left
    keyboard.Key.right: 4,   # Strong Right
    'd': 6,                  # Light Right
    keyboard.Key.down: 7,    # Strong Brake
    keyboard.Key.space: 8,   # Light Brake
    # No key press defaults to action 9 (Do Nothing)
}
QUIT_KEY = keyboard.Key.esc # Use Escape key to quit
# --- Key Mapping ---

# --- Global state for pressed keys and quit flag ---
pressed_keys = set()
quit_flag = threading.Event()
listener_thread = None
# --- Global state ---

# --- pynput Listener Callbacks (on_press, on_release) ---
# (Keep these functions as they were)
def on_press(key):
    global pressed_keys
    try:
        key_val = key.char if hasattr(key, 'char') else key
        if key_val in key_to_action_idx or key == QUIT_KEY:
            pressed_keys.add(key_val)
    except AttributeError: pass
    if key == QUIT_KEY:
        print("\nQuit key pressed!")
        quit_flag.set()
        return False

def on_release(key):
    global pressed_keys
    try:
        key_val = key.char if hasattr(key, 'char') else key
        if key_val in pressed_keys:
            pressed_keys.remove(key_val)
    except (AttributeError, KeyError): pass
# --- pynput Listener Callbacks ---

# --- get_human_action_idx, start_listener, stop_listener ---
# (Keep these functions as they were)
def get_human_action_idx():
    global pressed_keys
    current_pressed_indices = []
    for key_val, action_idx in key_to_action_idx.items():
         if key_val in pressed_keys: current_pressed_indices.append(action_idx)
    if not current_pressed_indices: return 9
    elif len(current_pressed_indices) == 1: return current_pressed_indices[0]
    else: # Handle multiple key presses
        if 3 in current_pressed_indices: return 3
        if 4 in current_pressed_indices: return 4
        if 5 in current_pressed_indices: return 5
        if 6 in current_pressed_indices: return 6
        if 0 in current_pressed_indices: return 0
        if 1 in current_pressed_indices: return 1
        if 2 in current_pressed_indices: return 2
        return current_pressed_indices[0]

def start_listener():
    global listener_thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener_thread = threading.Thread(target=listener.start, daemon=True)
    listener_thread.start()
    print("Keyboard listener started.")

def stop_listener():
    global listener_thread
    if listener_thread and listener_thread.is_alive():
         print("Waiting for listener thread to stop...")
         listener_thread.join(timeout=1.0)
    print("Keyboard listener stopped.")
# --- get_human_action_idx, start_listener, stop_listener ---


def collect_demonstrations(num_episodes_to_save=5): # Renamed parameter
    """Runs the environment interactively and collects human demonstrations, asking for confirmation."""

    print("\n--- Starting Human Demonstration Collection ---")
    print("Controls:")
    for key, idx in key_to_action_idx.items():
        key_name = key if isinstance(key, str) else key.name.upper()
        print(f"  {key_name.upper()}: Action {idx} -> {ACTION_SPACE[idx]}")
    print("  No Key Press: Action 9 -> [0. 0. 0.]")
    print(f"\nPress '{QUIT_KEY.name.upper()}' during an episode to quit the collection process.")
    print("-----------------------------------------------\n")

    start_listener()

    env_raw = gym.make(ENV_ID, render_mode='human', continuous=True)
    dummy_experiment = {'name': 'data_collection', 'record_video': -1}
    env = wrap_env(env_raw, dummy_experiment['name'], record_t=-1)

    all_saved_demonstrations = []
    saved_episodes_count = 0
    attempt_count = 0
    start_skip = 50

    try:
        # Loop until the desired number of episodes are SAVED
        while saved_episodes_count < num_episodes_to_save:
            if quit_flag.is_set():
                print("\nQuit signal received before starting new episode.")
                break

            attempt_count += 1
            print(f"\n--- Starting attempt {attempt_count} (Aiming for saved episode {saved_episodes_count + 1}/{num_episodes_to_save}) ---")
            state, info = env.reset()
            terminated = truncated = False
            step = 0
            current_episode_data = [] # Store data for this attempt

            # --- Run one episode attempt ---
            while not (terminated or truncated):
                if quit_flag.is_set():
                    print("\nQuit signal received mid-episode.")
                    break # Exit inner loop

                action_idx = get_human_action_idx()
                action_vector = ACTION_SPACE[action_idx]

                next_state, reward, terminated, truncated, info = env.step(action_vector)
                step += 1

                if step > start_skip:
                    state_cpu = state.cpu() if isinstance(state, torch.Tensor) and state.is_cuda else state
                    current_episode_data.append((state_cpu, action_idx))

                state = next_state
                time.sleep(0.02)
            # --- End of episode attempt ---

            # If quit signal received during episode, break outer loop too
            if quit_flag.is_set():
                break

            # --- Ask for confirmation ---
            print(f"--- Attempt {attempt_count} finished. Steps recorded: {len(current_episode_data)} ---")
            while True:
                save_choice = input("Save this episode? (y/n): ").strip().lower()
                if save_choice == 'y':
                    all_saved_demonstrations.extend(current_episode_data)
                    saved_episodes_count += 1
                    print(f"Episode saved. ({saved_episodes_count}/{num_episodes_to_save} saved)")
                    break
                elif save_choice == 'n':
                    print("Episode discarded.")
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
            # --- End of confirmation ---

    finally:
        print("\nClosing environment...")
        env.close()
        stop_listener()

    print(f"\n--- Collection Finished ---")
    print(f"Total episodes saved: {saved_episodes_count}")
    print(f"Total demonstrations saved: {len(all_saved_demonstrations)}")
    return all_saved_demonstrations


if __name__ == "__main__":
    print("Ensure the terminal or the game window has focus to capture keys...")

    existing_data = []
    mode = ''
    num_episodes_to_save = 10 # Default value

    # --- Ask user whether to overwrite or append ---
    if os.path.exists(OUTPUT_FILENAME):
        while True:
            print(f"\nExisting demonstration file found: '{OUTPUT_FILENAME}'")
            print("  1: Overwrite existing file and start fresh.")
            print("  2: Append new demonstrations to the existing file.")
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == '1':
                print("Selected: Overwrite existing demonstrations.")
                mode = 'overwrite'
                break
            elif choice == '2':
                print("Selected: Append to existing demonstrations.")
                mode = 'append'
                try:
                    with open(OUTPUT_FILENAME, 'rb') as f:
                        existing_data = pickle.load(f)
                    print(f"Loaded {len(existing_data)} existing demonstrations.")
                except Exception as e:
                    print(f"Error loading existing demonstrations: {e}. Starting fresh instead.")
                    existing_data = []
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    else:
        print(f"No existing demonstration file found. Starting fresh.")
        mode = 'overwrite'
    # --- Ask user whether to overwrite or append ---

    # --- Ask user how many episodes to SAVE ---
    while True:
        try:
            num_str = input(f"How many episodes do you want to SAVE this session? (default: {num_episodes_to_save}): ").strip()
            if not num_str: break
            num_episodes_to_save = int(num_str)
            if num_episodes_to_save > 0: break
            else: print("Please enter a positive number of episodes.")
        except ValueError: print("Invalid input. Please enter a number.")
    # --- Ask user how many episodes to SAVE ---


    # Collect new data, asking for confirmation after each episode attempt
    newly_collected_data = collect_demonstrations(num_episodes_to_save=num_episodes_to_save)

    # Combine data if appending
    if mode == 'append':
        final_data = existing_data + newly_collected_data
        print(f"Appended {len(newly_collected_data)} new demonstrations to existing data.")
    else: # Overwrite mode or starting fresh
        final_data = newly_collected_data

    # Save the final data
    if final_data:
        output_dir = os.path.dirname(OUTPUT_FILENAME)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        try:
            with open(OUTPUT_FILENAME, 'wb') as f: pickle.dump(final_data, f)
            print(f"\nSuccessfully saved {len(final_data)} total state-action pairs to {OUTPUT_FILENAME}")
        except Exception as e: print(f"\nError saving demonstrations: {e}")
    elif mode == 'overwrite' and os.path.exists(OUTPUT_FILENAME):
         print("\nNo new data collected/saved in overwrite mode. Removing existing file.")
         os.remove(OUTPUT_FILENAME)
    else:
        print("\nNo data collected or saved.")


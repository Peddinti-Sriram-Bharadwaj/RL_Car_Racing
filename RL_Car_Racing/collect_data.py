# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/collect_data.py
import gymnasium as gym
import numpy as np
# import keyboard  # Remove keyboard import
from pynput import keyboard # Import pynput instead
import time
import pickle
import os
import torch
import threading # Needed for the listener

# --- Important: Make sure these match your dqn.py and utils.py ---
from RL_Car_Racing.models.dqn import ACTION_SPACE
from RL_Car_Racing.utils import wrap_env
# --- Important: Make sure these match your dqn.py and utils.py ---

# --- Configuration ---
OUTPUT_FILENAME = "human_demonstrations.pkl"
ENV_ID = "CarRacing-v3" # Changed back to v2 for consistency
# --- Configuration ---

# --- Key Mapping (Using pynput Key objects and characters) ---
# Note: pynput uses specific objects for special keys (Key.up, Key.space)
# and character strings for regular keys ('w', 'a', 's', 'd').

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
quit_flag = threading.Event() # Use an Event for thread-safe quitting
listener_thread = None
# --- Global state ---

# --- pynput Listener Callbacks ---
def on_press(key):
    """Callback function for key press events."""
    global pressed_keys
    try:
        # Check if it's a character key or special key
        key_val = key.char if hasattr(key, 'char') else key
        if key_val in key_to_action_idx or key == QUIT_KEY:
            pressed_keys.add(key_val)
            # print(f"Pressed: {key_val}, Current set: {pressed_keys}") # Debug print
    except AttributeError:
        # Ignore keys that don't have a char attribute (like Shift, Ctrl) if not mapped
        pass
    if key == QUIT_KEY:
        print("Quit key pressed!")
        quit_flag.set() # Signal the main thread to quit
        return False # Stop the listener

def on_release(key):
    """Callback function for key release events."""
    global pressed_keys
    try:
        key_val = key.char if hasattr(key, 'char') else key
        if key_val in pressed_keys:
            pressed_keys.remove(key_val)
            # print(f"Released: {key_val}, Current set: {pressed_keys}") # Debug print
    except (AttributeError, KeyError):
        pass # Ignore if key wasn't tracked or doesn't have char

# --- pynput Listener Callbacks ---


def get_human_action_idx():
    """Checks the global set of pressed keys and returns the corresponding action index."""
    global pressed_keys
    current_pressed_indices = []

    # Check the currently pressed keys against our mapping
    # Need to iterate through the mapping, not the pressed_keys set directly
    # because the set might contain keys not in our mapping (like Shift)
    for key_val, action_idx in key_to_action_idx.items():
         if key_val in pressed_keys:
              current_pressed_indices.append(action_idx)

    if not current_pressed_indices:
        return 9 # Default to "Do Nothing"
    elif len(current_pressed_indices) == 1:
        return current_pressed_indices[0]
    else:
        # Handle multiple key presses (same priority logic as before)
        if 3 in current_pressed_indices: return 3
        if 4 in current_pressed_indices: return 4
        if 5 in current_pressed_indices: return 5
        if 6 in current_pressed_indices: return 6
        if 0 in current_pressed_indices: return 0
        if 1 in current_pressed_indices: return 1
        if 2 in current_pressed_indices: return 2
        return current_pressed_indices[0]


def start_listener():
    """Starts the pynput keyboard listener in a separate thread."""
    global listener_thread
    # Setup the listener (non-blocking)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener_thread = threading.Thread(target=listener.start, daemon=True)
    listener_thread.start()
    print("Keyboard listener started.")

def stop_listener():
    """Stops the pynput keyboard listener."""
    global listener_thread
    # The listener stops itself when QUIT_KEY is pressed (on_press returns False)
    # We just need to wait for the thread to finish if it was started
    if listener_thread and listener_thread.is_alive():
         print("Waiting for listener thread to stop...")
         listener_thread.join(timeout=1.0) # Wait briefly
    print("Keyboard listener stopped.")


def collect_demonstrations(num_episodes=5):
    """Runs the environment interactively and collects human demonstrations."""
    print("\n--- Starting Human Demonstration Collection ---")
    print("Controls:")
    for key, idx in key_to_action_idx.items():
        key_name = key if isinstance(key, str) else key.name.upper()
        print(f"  {key_name.upper()}: Action {idx} -> {ACTION_SPACE[idx]}")
    print("  No Key Press: Action 9 -> [0. 0. 0.]")
    print(f"\nPress '{QUIT_KEY.name.upper()}' to quit early.") # Use QUIT_KEY name
    print("-----------------------------------------------\n")

    start_listener() # Start listening for keys

    env_raw = gym.make(ENV_ID, render_mode='human', continuous=True)
    dummy_experiment = {'name': 'data_collection', 'record_video': -1}
    env = wrap_env(env_raw, dummy_experiment['name'], record_t=-1)

    demonstrations = []
    start_skip = 50

    try: # Use try...finally to ensure listener stops
        for episode in range(num_episodes):
            if quit_flag.is_set(): break # Check if quit was signalled

            print(f"Starting Episode {episode + 1}/{num_episodes}")
            state, info = env.reset()
            terminated = truncated = False
            step = 0

            while not (terminated or truncated):
                if quit_flag.is_set(): # Check frequently within the episode
                    print("Quitting data collection...")
                    break

                action_idx = get_human_action_idx()
                action_vector = ACTION_SPACE[action_idx]

                next_state, reward, terminated, truncated, info = env.step(action_vector)
                step += 1

                if step > start_skip:
                    state_cpu = state.cpu() if isinstance(state, torch.Tensor) and state.is_cuda else state
                    demonstrations.append((state_cpu, action_idx))

                state = next_state
                time.sleep(0.02) # Keep the small delay

            if quit_flag.is_set(): break # Exit outer loop if quit signal received
            print(f"Episode {episode + 1} finished. Total demonstrations collected: {len(demonstrations)}")

    finally: # Ensure cleanup happens
        print("Closing environment...")
        env.close()
        stop_listener() # Stop the listener thread

    return demonstrations

if __name__ == "__main__":
    print("Ensure the terminal or the game window has focus to capture keys...")
    # No need for extra sleep, listener starts immediately

    collected_data = collect_demonstrations(num_episodes=10)

    if collected_data:
        output_dir = os.path.dirname(OUTPUT_FILENAME)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(OUTPUT_FILENAME, 'wb') as f:
            pickle.dump(collected_data, f)
        print(f"\nSuccessfully saved {len(collected_data)} state-action pairs to {OUTPUT_FILENAME}")
    else:
        print("\nNo data collected or collection quit early.")


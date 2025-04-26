# /Users/srirambharadwaj/Documents/iiitb/sem2/RL/RL_Car_Racing/RL_Car_Racing/train_imitation.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np

# --- Important: Make sure these match your dqn.py ---
from RL_Car_Racing.models.dqn import QNetwork, ACTION_SPACE
# --- Important: Make sure these match your dqn.py ---

# --- Configuration ---
DATA_FILENAME = "human_demonstrations.pkl"
MODEL_SAVE_PATH = "imitation_model.pth"
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
# --- Configuration ---

class DemonstrationDataset(Dataset):
    """PyTorch Dataset for loading human demonstrations."""
    def __init__(self, data):
        self.data = data # List of (state_tensor, action_index) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action_idx = self.data[idx]
        # Ensure state is a tensor (it should be if saved correctly)
        if not isinstance(state, torch.Tensor):
             state = torch.tensor(state, dtype=torch.float32)
        # Action index should be a LongTensor for CrossEntropyLoss
        action_idx = torch.tensor(action_idx, dtype=torch.long)
        return state, action_idx

def train_behavioral_cloning():
    """Loads data and trains the network using behavioral cloning."""
    print(f"--- Starting Behavioral Cloning Training ---")

    # --- Load Data ---
    if not os.path.exists(DATA_FILENAME):
        print(f"Error: Demonstration file not found at {DATA_FILENAME}")
        print("Please run collect_data.py first.")
        return

    print(f"Loading data from {DATA_FILENAME}...")
    with open(DATA_FILENAME, 'rb') as f:
        demonstrations = pickle.load(f)
    print(f"Loaded {len(demonstrations)} demonstrations.")

    if not demonstrations:
        print("Error: No demonstrations found in the data file.")
        return
    # --- Load Data ---

    # --- Setup Model, Optimizer, Loss ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_actions = len(ACTION_SPACE)
    # Get input channels from the state tensor shape (assuming CHW)
    input_channels = demonstrations[0][0].shape[0]
    model = QNetwork(num_actions=num_actions, input_channels=input_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Use CrossEntropyLoss for classification (predicting action index)
    criterion = nn.CrossEntropyLoss()

    # --- Setup DataLoader ---
    dataset = DemonstrationDataset(demonstrations)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) # Use num_workers > 0 if possible
    print(f"Dataset size: {len(dataset)}, DataLoader batches: {len(dataloader)}")
    # --- Setup DataLoader ---

    # --- Training Loop ---
    print(f"Training for {NUM_EPOCHS} epochs...")
    model.train() # Set model to training mode
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (states, target_actions) in enumerate(dataloader):
            states = states.to(device)
            target_actions = target_actions.to(device) # Target actions are indices (LongTensor)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass - output are logits for each action
            output_logits = model(states)

            # Calculate loss
            loss = criterion(output_logits, target_actions)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # --- Track metrics ---
            total_loss += loss.item()
            # Calculate accuracy
            _, predicted_actions = torch.max(output_logits, 1)
            correct_predictions += (predicted_actions == target_actions).sum().item()
            total_samples += target_actions.size(0)
            # --- Track metrics ---

            if (batch_idx + 1) % 50 == 0: # Print progress every 50 batches
                 print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")


        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Complete. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    # --- Training Loop ---

    # --- Save Model ---
    print(f"\nTraining finished. Saving model to {MODEL_SAVE_PATH}...")
    # Ensure directory exists
    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")
    # --- Save Model ---

if __name__ == "__main__":
    train_behavioral_cloning()

"""
Neural Network for Cube Solving

A simple feedforward network that takes the cube state as input
and outputs action probabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CubeSolverNetwork(nn.Module):
    """
    Neural network for predicting cube-solving moves.

    Architecture:
        Input: 324 (54 positions × 6 colors one-hot)
        Hidden1: 256 neurons with ReLU
        Hidden2: 128 neurons with ReLU
        Output: 18 (probability for each move)
    """

    def __init__(self, hidden1: int = 256, hidden2: int = 128):
        super().__init__()

        self.input_size = 324  # 54 * 6
        self.output_size = 18  # Number of moves

        self.fc1 = nn.Linear(self.input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, self.output_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 324)

        Returns:
            Output tensor of shape (batch, 18) - action probabilities
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

    def predict_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Predict the best action for a given state.

        Args:
            state: One-hot encoded cube state (324,)
            deterministic: If True, always pick highest probability.
                          If False, sample from distribution.

        Returns:
            Action index (0-17)
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.forward(state_tensor).squeeze().numpy()

            if deterministic:
                return int(np.argmax(probs))
            else:
                # Sample from probability distribution
                return int(np.random.choice(len(probs), p=probs))

    def get_weights_flat(self) -> np.ndarray:
        """Get all weights as a flat numpy array."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights_flat(self, flat_weights: np.ndarray):
        """Set weights from a flat numpy array."""
        idx = 0
        for param in self.parameters():
            param_shape = param.shape
            param_size = np.prod(param_shape)
            param.data = torch.FloatTensor(
                flat_weights[idx : idx + param_size].reshape(param_shape)
            )
            idx += param_size

    def get_weight_count(self) -> int:
        """Get total number of weights in the network."""
        return sum(p.numel() for p in self.parameters())

    def get_architecture(self) -> dict:
        """Get network architecture as a dictionary."""
        return {
            "input_size": self.input_size,
            "hidden1": self.fc1.out_features,
            "hidden2": self.fc2.out_features,
            "output_size": self.output_size,
            "total_weights": self.get_weight_count(),
        }


def create_network_from_weights(
    weights: np.ndarray, hidden1: int = 256, hidden2: int = 128
) -> CubeSolverNetwork:
    """
    Create a network and load weights.

    Args:
        weights: Flat array of weights
        hidden1: First hidden layer size
        hidden2: Second hidden layer size

    Returns:
        Initialized network
    """
    network = CubeSolverNetwork(hidden1, hidden2)
    network.set_weights_flat(weights)
    return network

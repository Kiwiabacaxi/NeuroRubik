"""
Neural Network for Cube Solving - GPU Enhanced Version

A deeper feedforward network optimized for GPU training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CubeSolverNetwork(nn.Module):
    """
    Enhanced neural network for predicting cube-solving moves.

    Architecture:
        Input: 324 (54 positions × 6 colors one-hot)
        Hidden layers: Configurable depth and width
        Output: 18 (probability for each move)
    """

    def __init__(
        self, 
        hidden_sizes: tuple = (512, 512, 256, 128),
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.device = device or DEVICE
        self.input_size = 324  # 54 * 6
        self.output_size = 18  # Number of moves
        
        # Build layers dynamically
        layers = []
        prev_size = self.input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0 and i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.network = nn.Sequential(*layers)
        self.hidden_sizes = hidden_sizes
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)

    def _init_weights(self):
        """Initialize weights with He initialization for ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 324)

        Returns:
            Output tensor of shape (batch, 18) - action logits
        """
        x = self.network(x)
        return F.softmax(x, dim=-1)

    def predict_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict the best action for a given state.

        Args:
            state: One-hot encoded cube state (324,)
            deterministic: If True, always pick highest probability.

        Returns:
            Action index (0-17)
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs = self.forward(state_tensor).squeeze().cpu().numpy()

            if deterministic:
                return int(np.argmax(probs))
            else:
                return int(np.random.choice(len(probs), p=probs))

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Predict actions for a batch of states (GPU accelerated).

        Args:
            states: Array of shape (batch, 324)

        Returns:
            Array of action indices (batch,)
        """
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            probs = self.forward(states_tensor).cpu().numpy()
            return np.argmax(probs, axis=1)

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
            param_size = int(np.prod(param_shape))
            new_data = torch.FloatTensor(
                flat_weights[idx : idx + param_size].reshape(param_shape)
            ).to(self.device)
            param.data.copy_(new_data)
            idx += param_size

    def get_weight_count(self) -> int:
        """Get total number of weights in the network."""
        return sum(p.numel() for p in self.parameters())

    def get_architecture(self) -> dict:
        """Get network architecture as a dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "total_weights": self.get_weight_count(),
            "device": str(self.device)
        }


def create_network_from_weights(
    weights: np.ndarray, 
    hidden_sizes: tuple = (512, 512, 256, 128),
    device: Optional[torch.device] = None
) -> CubeSolverNetwork:
    """
    Create a network and load weights.

    Args:
        weights: Flat array of weights
        hidden_sizes: Hidden layer sizes

    Returns:
        Initialized network
    """
    network = CubeSolverNetwork(hidden_sizes=hidden_sizes, device=device)
    network.set_weights_flat(weights)
    return network


def get_device_info() -> dict:
    """Get information about available compute devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info

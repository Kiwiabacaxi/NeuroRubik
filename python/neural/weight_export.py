"""
Weight Export/Import for TensorFlow.js Compatibility

Exports PyTorch weights to JSON format that can be loaded by TensorFlow.js
in the frontend.
"""

import json
from typing import Any, Dict

import numpy as np

from .network import CubeSolverNetwork


def export_weights_to_json(network: CubeSolverNetwork, filepath: str):
    """
    Export network weights to JSON for TensorFlow.js.

    The format is compatible with loading in the browser:
    {
        "architecture": {...},
        "layers": [
            {"name": "fc1", "weights": [...], "bias": [...], "shape": [...]},
            ...
        ]
    }
    """
    data = {"architecture": network.get_architecture(), "layers": []}

    # Export each layer
    layer_names = ["fc1", "fc2", "fc3"]
    layers = [network.fc1, network.fc2, network.fc3]

    for name, layer in zip(layer_names, layers):
        layer_data = {
            "name": name,
            "weights": layer.weight.data.cpu().numpy().tolist(),
            "bias": layer.bias.data.cpu().numpy().tolist(),
            "weight_shape": list(layer.weight.shape),
            "bias_shape": list(layer.bias.shape),
        }
        data["layers"].append(layer_data)

    with open(filepath, "w") as f:
        json.dump(data, f)

    print(f"Weights exported to {filepath}")


def load_weights_from_json(network: CubeSolverNetwork, filepath: str):
    """
    Load weights from JSON file.

    Args:
        network: Network to load weights into
        filepath: Path to JSON file
    """
    import torch

    with open(filepath) as f:
        data = json.load(f)

    layers = [network.fc1, network.fc2, network.fc3]

    for layer_data, layer in zip(data["layers"], layers):
        weights = np.array(layer_data["weights"], dtype=np.float32)
        bias = np.array(layer_data["bias"], dtype=np.float32)

        layer.weight.data = torch.FloatTensor(weights)
        layer.bias.data = torch.FloatTensor(bias)

    print(f"Weights loaded from {filepath}")


def export_training_stats(stats: Dict[str, Any], filepath: str):
    """Export training statistics to JSON."""

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(stats), f, indent=2)

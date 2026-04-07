"""
Policy Network

Neural network architecture for policy-based RL agents.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np


class PolicyNetwork:
    """
    Neural network for learning policies.
    
    Lightweight NumPy MLP implementation for inference-side policy/value calls.
    
    Attributes:
        input_dim: Input dimension (observation space)
        output_dim: Output dimension (action space)
        hidden_dims: List of hidden layer dimensions
        architecture: Network architecture type
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize policy network.
        
        Args:
            config: Configuration with network architecture parameters
        """
        self.config = config
        self.input_dim = config.get("input_dim", 128)
        self.output_dim = config.get("output_dim", 10)
        self.hidden_dims = config.get("hidden_dims", [256, 256])
        self.architecture = config.get("architecture", "mlp")
        self._build_network()
    
    def _build_network(self) -> None:
        """Build the neural network architecture."""
        if self.architecture != "mlp":
            raise ValueError(f"Unsupported architecture: {self.architecture}. Only 'mlp' is supported.")

        # Deterministic initialization for reproducibility.
        seed = int(self.config.get("seed", 42))
        rng = np.random.default_rng(seed)

        dims = [self.input_dim] + list(self.hidden_dims)
        self.hidden_weights: List[np.ndarray] = []
        self.hidden_biases: List[np.ndarray] = []

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            # Xavier-like scaling for stable outputs.
            scale = np.sqrt(2.0 / max(d_in + d_out, 1))
            self.hidden_weights.append((rng.standard_normal((d_in, d_out)) * scale).astype(np.float32))
            self.hidden_biases.append(np.zeros((d_out,), dtype=np.float32))

        last_dim = dims[-1]
        self.policy_w = (rng.standard_normal((last_dim, self.output_dim)) * np.sqrt(2.0 / max(last_dim + self.output_dim, 1))).astype(np.float32)
        self.policy_b = np.zeros((self.output_dim,), dtype=np.float32)

        self.value_w = (rng.standard_normal((last_dim, 1)) * np.sqrt(2.0 / max(last_dim + 1, 1))).astype(np.float32)
        self.value_b = np.zeros((1,), dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _forward_hidden(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Observation dimension mismatch: expected {self.input_dim}, got {x.shape[0]}")
        for w, b in zip(self.hidden_weights, self.hidden_biases):
            x = self._relu(x @ w + b)
        return x

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        denom = np.sum(exp_z)
        if denom <= 0:
            return np.ones_like(logits, dtype=np.float32) / float(len(logits))
        return (exp_z / denom).astype(np.float32)
    
    def forward(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network.
        
        Args:
            observation: Input observation
            
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        h = self._forward_hidden(observation)
        logits = (h @ self.policy_w + self.policy_b).astype(np.float32)
        value = (h @ self.value_w + self.value_b).astype(np.float32)
        return logits, value
    
    def predict_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action given observation.
        
        Args:
            observation: Input observation
            
        Returns:
            Action probabilities or logits
        """
        probs = self.get_action_distribution(observation)
        return np.array([int(np.argmax(probs))], dtype=np.int64)
    
    def predict_value(self, observation: np.ndarray) -> float:
        """
        Predict value estimate for observation.
        
        Args:
            observation: Input observation
            
        Returns:
            Value estimate
        """
        _, value = self.forward(observation)
        return float(value.reshape(-1)[0])
    
    def get_action_distribution(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution.
        
        Args:
            observation: Input observation
            
        Returns:
            Action probability distribution
        """
        logits, _ = self.forward(observation)
        return self._softmax(logits)
    
    def save_weights(self, filepath: str) -> None:
        """
        Save network weights to file.
        
        Args:
            filepath: Path to save weights
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dim": np.array([self.input_dim], dtype=np.int32),
            "output_dim": np.array([self.output_dim], dtype=np.int32),
            "hidden_dims": np.asarray(self.hidden_dims, dtype=np.int32),
            "policy_w": self.policy_w,
            "policy_b": self.policy_b,
            "value_w": self.value_w,
            "value_b": self.value_b,
            "num_hidden": np.array([len(self.hidden_weights)], dtype=np.int32),
        }
        for idx, (w, b) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            payload[f"hidden_w_{idx}"] = w
            payload[f"hidden_b_{idx}"] = b
        np.savez(path, **payload)
    
    def load_weights(self, filepath: str) -> None:
        """
        Load network weights from file.
        
        Args:
            filepath: Path to load weights
        """
        with np.load(filepath, allow_pickle=False) as data:
            self.input_dim = int(data["input_dim"][0])
            self.output_dim = int(data["output_dim"][0])
            self.hidden_dims = [int(v) for v in data["hidden_dims"].tolist()]

            num_hidden = int(data["num_hidden"][0])
            self.hidden_weights = []
            self.hidden_biases = []
            for idx in range(num_hidden):
                self.hidden_weights.append(data[f"hidden_w_{idx}"].astype(np.float32))
                self.hidden_biases.append(data[f"hidden_b_{idx}"].astype(np.float32))

            self.policy_w = data["policy_w"].astype(np.float32)
            self.policy_b = data["policy_b"].astype(np.float32)
            self.value_w = data["value_w"].astype(np.float32)
            self.value_b = data["value_b"].astype(np.float32)
    
    def get_trainable_params(self) -> List:
        """
        Get list of trainable parameters.
        
        Returns:
            List of trainable parameters
        """
        params: List[np.ndarray] = []
        params.extend(self.hidden_weights)
        params.extend(self.hidden_biases)
        params.extend([self.policy_w, self.policy_b, self.value_w, self.value_b])
        return params

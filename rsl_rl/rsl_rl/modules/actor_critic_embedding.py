#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""
ActorCritic with Embedding Layer for Transformer Integration

Extends standard ActorCritic with 2-layer embedding MLP before actor/critic networks.
Outputs 512-dim feature vectors for transformer compatibility.

Architecture: Observations -> Embedding MLP -> 512-dim features -> Actor/Critic MLPs
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class EmbeddingMLP(nn.Module):
    """Two-layer MLP for embedding observations into fixed-size feature vectors.
    
    Args:
        input_dim (int): Input observation dimensions
        output_dim (int): Output feature dimensions (default: 512)
        hidden_dim (int): Hidden layer dimensions (default: 256)
        activation (str): Activation function (default: "elu")
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 256,
        activation: str = "elu",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        activation_fn = get_activation(activation)
        
        self.embedding_layers = nn.Sequential(
            # First layer: input -> hidden
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout),
            
            # Second layer: hidden -> output (512-dim token)
            nn.Linear(hidden_dim, output_dim),
            activation_fn,
        )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.embedding_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass: observations -> embedded features."""
        return self.embedding_layers(observations)


class ActorCriticEmbedding(nn.Module):
    """ActorCritic with embedding layer for transformer compatibility.
    
    Args:
        num_actor_obs (int): Actor observation dimensions
        num_critic_obs (int): Critic observation dimensions  
        num_actions (int): Action dimensions
        embedding_dim (int): Embedding feature dimensions (default: 512)
        embedding_hidden_dim (int): Embedding hidden dimensions (default: 256)
        actor_hidden_dims (list): Actor MLP hidden dimensions
        critic_hidden_dims (list): Critic MLP hidden dimensions
        activation (str): Activation function name
        init_noise_std (float): Initial action noise std
        embedding_dropout (float): Embedding dropout rate
    """
    
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        embedding_dim: int = 512,
        embedding_hidden_dim: int = 256,
        actor_hidden_dims: list = [256, 256, 256],
        critic_hidden_dims: list = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        embedding_dropout: float = 0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEmbedding.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        activation_fn = get_activation(activation)
        
        # Store dimensions
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        
        # Embedding layers for actor and critic observations
        self.actor_embedding = EmbeddingMLP(
            input_dim=num_actor_obs,
            output_dim=embedding_dim,
            hidden_dim=embedding_hidden_dim,
            activation=activation,
            dropout=embedding_dropout,
        )
        
        # If actor and critic have different observation dimensions, create separate embedding
        if num_actor_obs != num_critic_obs:
            self.critic_embedding = EmbeddingMLP(
                input_dim=num_critic_obs,
                output_dim=embedding_dim,
                hidden_dim=embedding_hidden_dim,
                activation=activation,
                dropout=embedding_dropout,
            )
        else:
            # Share embedding if observations are the same
            self.critic_embedding = self.actor_embedding
        
        # Actor network: embedding_dim -> actions
        actor_layers = []
        prev_dim = embedding_dim
        for i, hidden_dim in enumerate(actor_hidden_dims):
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn,
            ])
            prev_dim = hidden_dim
        # Final output layer
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network: embedding_dim -> value
        critic_layers = []
        prev_dim = embedding_dim
        for i, hidden_dim in enumerate(critic_hidden_dims):
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn,
            ])
            prev_dim = hidden_dim
        # Final output layer
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Embedding: {self.actor_embedding}")
        print(f"Critic Embedding: {self.critic_embedding}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        """Initialize weights for sequential layers."""
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """Reset method for compatibility."""
        pass

    def forward(self):
        """Use specific methods instead."""
        raise NotImplementedError

    def get_actor_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get actor embedding features."""
        return self.actor_embedding(observations)

    def get_critic_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get critic embedding features."""
        return self.critic_embedding(observations)

    @property
    def action_mean(self):
        """Mean of the action distribution."""
        return self.distribution.mean

    @property
    def action_std(self):
        """Standard deviation of the action distribution."""
        return self.distribution.stddev

    @property
    def entropy(self):
        """Entropy of the action distribution."""
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        """Update action distribution based on observations."""
        embedded_features = self.actor_embedding(observations)
        mean = self.actor(embedded_features)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample actions from policy."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action selection (mean)."""
        embedded_features = self.actor_embedding(observations)
        actions_mean = self.actor(embedded_features)
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate state value using critic network."""
        embedded_features = self.critic_embedding(critic_observations)
        value = self.critic(embedded_features)
        return value


def get_activation(act_name: str) -> nn.Module:
    """Get activation function by name."""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()  # CReLU is not available in standard PyTorch
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print(f"Invalid activation function: {act_name}! Using ReLU as default.")
        return nn.ReLU()
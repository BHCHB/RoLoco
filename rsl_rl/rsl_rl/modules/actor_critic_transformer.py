#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""
ActorCritic with Transformer Architecture

Implements ActorCritic with Transformer backbone using cusrl's transformer module.
- Embedding layer: Observations -> 256-dim token
- Transformer: cusrl TransformerEncoderLayer (6 layers, hidden 256, multi-head attention)
- Actor/Critic        # Transformer backbone for sequential processing
        # Will automatically use cusrl if available, otherwise fallback to basic PyTorch
        self.transformer = TransformerBackbone(
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            feedforward_dim=transformer_feedforward_dim,
            dropout=transformer_dropout,
            segment_length=segment_length,
            use_cusrl=True,  # Enable cusrl if available
        )ocess transformer output

Architecture: Observations -> Embedding -> cusrl Transformer -> Actor/Critic MLPs
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import sys
import os

# Add cusrl to path if not already there
cusrl_path = "/home/zju/Documents/20251107loco/cusrl"
if cusrl_path not in sys.path:
    sys.path.insert(0, cusrl_path)

# Import cusrl transformer module
try:
    from cusrl.module.transformer import TransformerEncoderLayer
    from cusrl.module.causal_attn import CausalTransformerEncoderLayer
    CUSRL_AVAILABLE = True
    CAUSAL_AVAILABLE = True
    print("[INFO] Successfully imported cusrl transformer modules (standard + causal)")
except ImportError as e:
    print(f"[WARNING] Could not import cusrl transformer: {e}")
    print("[INFO] Will use basic PyTorch MultiheadAttention fallback")
    CUSRL_AVAILABLE = False
    CAUSAL_AVAILABLE = False


class EmbeddingMLP(nn.Module):
    """Two-layer MLP for embedding observations into fixed-size feature vectors.
    
    Args:
        input_dim (int): Input observation dimensions
        output_dim (int): Output feature dimensions (default: 256 for transformer)
        hidden_dim (int): Hidden layer dimensions (default: 256)
        activation (str): Activation function (default: "elu")
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
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
            
            # Second layer: hidden -> output (256-dim token)
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


class TransformerBackbone(nn.Module):
    """Transformer backbone using cusrl's CausalTransformerEncoderLayer with RoPE and memory.
    
    Args:
        embed_dim (int): Embedding dimension (default: 256)
        num_layers (int): Number of transformer layers (default: 6)
        num_heads (int): Number of attention heads (default: 4)
        feedforward_dim (int): Feed-forward hidden dimension (default: 512)
        dropout (float): Dropout rate (default: 0.1)
        window_size (int): Causal attention window size for memory (default: 128)
        rope_base (float): RoPE base frequency (default: 10000.0)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        window_size: int = 128,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        
        if not CAUSAL_AVAILABLE:
            raise ImportError(
                "CausalTransformerEncoderLayer not available. Please ensure cusrl and FlashAttention are installed."
            )
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.window_size = window_size
        
        print(f"[INFO] Initializing CausalTransformerEncoderLayer with {num_layers} layers, RoPE base={rope_base}, window={window_size}")
        
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                feedforward_dim=feedforward_dim,
                activation_fn=nn.GELU,
                dropout=dropout,
                dtype=torch.float16,  # FlashAttention requires float16 or bfloat16
                gate_type="residual",
                layer_norm="post",
                use_alibi=False,  # Use RoPE instead of ALiBi
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        ])
        
        print(f"[INFO] ✓ Successfully initialized CausalTransformerEncoderLayer with RoPE and memory")
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Memory for each layer (list of tuples)
        self.hidden_states = None
        
    def get_hidden_states(self):
        """Get current hidden states (memory)."""
        return self.hidden_states
    
    def reset(self, dones=None):
        """Reset memory for done environments."""
        if self.hidden_states is None:
            return
        
        if dones is None:
            # Reset all memory
            self.hidden_states = None
        else:
            # Reset only done environments for each layer
            for layer_idx, layer in enumerate(self.layers):
                if self.hidden_states[layer_idx] is not None:
                    layer.reset_memory(self.hidden_states[layer_idx], done=dones)
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden_states: list = None,
        done: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list]:
        """Forward pass through causal transformer layers.
        
        Args:
            x: Input tensor of shape (batch, embed_dim)
            hidden_states: Previous hidden states (list of tuples, one per layer)
            done: Done mask for resetting memory (batch, 1) boolean tensor
            
        Returns:
            output: Processed features (batch, embed_dim)
            new_hidden_states: Updated hidden states for next iteration
        """
        # Causal Transformer with memory (seq_first format: L, N, C)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (batch, dim) -> (1, batch, dim)
        elif x.dim() == 3:
            x = x.transpose(0, 1)  # (batch, seq, dim) -> (seq, batch, dim)
        
        # Initialize hidden states if needed
        if hidden_states is None:
            hidden_states = [None] * len(self.layers)
        
        # Pass through causal transformer layers
        new_hidden_states = []
        for i, layer in enumerate(self.layers):
            x, new_mem = layer(
                x,
                memory=hidden_states[i],
                done=done,
                sequential=True,
            )
            new_hidden_states.append(new_mem)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Extract output: take last token (seq, batch, dim) -> (batch, dim)
        output = x[-1, :, :]
        
        return output, new_hidden_states


class BasicTransformerEncoderLayer(nn.Module):
    """Fallback basic Transformer Encoder Layer when cusrl is not available."""
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        activation_fn = get_activation(activation)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass with post-norm architecture."""
        # Attention + residual
        src2, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            is_causal=is_causal,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward + residual
        src2 = self.feedforward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class ActorCriticTransformer(nn.Module):
    """ActorCritic with Causal Transformer backbone (RoPE + Memory).
    
    Architecture:
    1. Embedding: Observations -> 256-dim token (2-layer MLP)
    2. Causal Transformer: 6 layers with RoPE, multi-head attention, and KV cache memory
    3. Actor/Critic MLPs: Process transformer output -> actions/value
    
    Args:
        num_actor_obs (int): Actor observation dimensions
        num_critic_obs (int): Critic observation dimensions  
        num_actions (int): Action dimensions
        embedding_dim (int): Embedding/Transformer dimension (default: 256)
        embedding_hidden_dim (int): Embedding MLP hidden dimension (default: 256)
        transformer_layers (int): Number of transformer layers (default: 6)
        transformer_heads (int): Number of attention heads (default: 4)
        transformer_feedforward_dim (int): Transformer FFN dimension (default: 512)
        window_size (int): Causal attention window size for memory (default: 128)
        actor_hidden_dims (list): Actor MLP hidden dimensions
        critic_hidden_dims (list): Critic MLP hidden dimensions
        activation (str): Activation function name
        init_noise_std (float): Initial action noise std
        embedding_dropout (float): Embedding dropout rate
        transformer_dropout (float): Transformer dropout rate
        rope_base (float): RoPE base frequency (default: 10000.0)
    """
    
    # Always recurrent (has memory)
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        embedding_dim: int = 256,
        embedding_hidden_dim: int = 256,
        transformer_layers: int = 6,
        transformer_heads: int = 4,
        transformer_feedforward_dim: int = 512,
        window_size: int = 128,
        actor_hidden_dims: list = [256, 256, 128],
        critic_hidden_dims: list = [256, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        embedding_dropout: float = 0.1,
        transformer_dropout: float = 0.1,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        activation_fn = get_activation(activation)
        
        # Store dimensions
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        
        # 1. Shared Embedding layer: Observations -> 256-dim tokens (2-layer MLP)
        # Actor and critic ALWAYS share the same embedding when obs dims are same
        if num_actor_obs != num_critic_obs:
            raise ValueError(
                f"ActorCriticTransformer requires same observation dimensions for actor and critic. "
                f"Got num_actor_obs={num_actor_obs}, num_critic_obs={num_critic_obs}. "
                f"Use privileged observations in actor_obs only, not critic_obs."
            )
        
        self.shared_embedding = EmbeddingMLP(
            input_dim=num_actor_obs,
            output_dim=embedding_dim,
            hidden_dim=embedding_hidden_dim,
            activation=activation,
            dropout=embedding_dropout,
        )
        
        # 2. Shared Causal Transformer backbone (with RoPE and memory)
        self.transformer = TransformerBackbone(
            embed_dim=embedding_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            feedforward_dim=transformer_feedforward_dim,
            dropout=transformer_dropout,
            window_size=window_size,
            rope_base=rope_base,
        )
        
        # 3. Actor MLP: transformer output -> actions
        actor_layers = []
        actor_layers.append(nn.Linear(embedding_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # 4. Critic MLP: transformer output -> value
        critic_layers = []
        critic_layers.append(nn.Linear(embedding_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        print(f"=== ActorCriticTransformer Architecture ===")
        print(f"Shared Embedding (2-layer MLP): {self.shared_embedding}")
        print(f"Shared Causal Transformer: {transformer_layers} layers, {transformer_heads} heads, dim={embedding_dim}, FFN={transformer_feedforward_dim}")
        print(f"RoPE base: {rope_base}, Window size: {window_size}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def get_hidden_states(self):
        """Get hidden states for recurrent policy (similar to ActorCriticRecurrent)."""
        # Return tuple (actor_hidden, critic_hidden)
        # Actor and critic share the same transformer, so they share hidden states
        hid = self.transformer.get_hidden_states()
        return (hid, hid)

    def reset(self, dones=None):
        """Reset transformer memory for terminated episodes."""
        self.transformer.reset(dones)

    def forward(self):
        """Use specific methods instead."""
        raise NotImplementedError

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

    def act(self, observations: torch.Tensor, masks=None, hidden_states=None, **kwargs) -> torch.Tensor:
        """Sample actions from policy (similar to ActorCriticRecurrent).
        
        Args:
            observations: Observation tensor
            masks: Optional mask tensor (not used, for compatibility)
            hidden_states: Optional hidden states from previous timestep (for batch training)
        """
        # Batch mode (policy update): use provided hidden states
        batch_mode = hidden_states is not None
        if batch_mode and hidden_states is None:
            raise ValueError("Hidden states not passed to transformer module during policy update")
        
        # 1. Embedding
        embedded = self.shared_embedding(observations)
        
        # 2. Transformer forward
        if batch_mode:
            # Use provided hidden states during training
            features, _ = self.transformer(embedded, hidden_states=hidden_states, done=None)
        else:
            # Use internal hidden states during rollout
            features, self.transformer.hidden_states = self.transformer(
                embedded, 
                hidden_states=self.transformer.hidden_states,
                done=None
            )
        
        # 3. Actor output
        mean = self.actor(features)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action selection (mean) for inference."""
        # 1. Embedding
        embedded = self.shared_embedding(observations)
        
        # 2. Transformer forward
        features, self.transformer.hidden_states = self.transformer(
            embedded,
            hidden_states=self.transformer.hidden_states,
            done=None,
        )
        
        # 3. Actor output -> deterministic actions
        actions_mean = self.actor(features)
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor, masks=None, hidden_states=None, **kwargs) -> torch.Tensor:
        """Evaluate state value using critic network (similar to ActorCriticRecurrent).
        
        Args:
            critic_observations: Critic observation tensor
            masks: Optional mask tensor (not used, for compatibility)
            hidden_states: Optional hidden states from previous timestep (for batch training)
        """
        # Batch mode (policy update): use provided hidden states
        batch_mode = hidden_states is not None
        if batch_mode and hidden_states is None:
            raise ValueError("Hidden states not passed to transformer module during policy update")
        
        # 1. Embedding
        embedded = self.shared_embedding(critic_observations)
        
        # 2. Transformer forward
        if batch_mode:
            # Use provided hidden states during training
            features, _ = self.transformer(embedded, hidden_states=hidden_states, done=None)
        else:
            # Use internal hidden states during rollout
            features, self.transformer.hidden_states = self.transformer(
                embedded,
                hidden_states=self.transformer.hidden_states,
                done=None
            )
        
        # 3. Critic output
        value = self.critic(features)
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
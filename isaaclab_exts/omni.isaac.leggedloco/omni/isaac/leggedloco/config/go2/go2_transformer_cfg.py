# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2 Configuration with Transformer-based ActorCritic

This configuration uses ActorCriticTransformer with:
- Embedding: Observations -> 256-dim token
- Transformer: 6 layers, 4 heads, hidden 256, FFN 512
- Segment length: 128 (Transformer-XL memory)
- Actor/Critic MLPs: Process transformer output
- NO privileged observations: Actor and Critic use identical observations
"""

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import omni.isaac.leggedloco.leggedloco.mdp as mdp

# Import base configuration
from .go2_low_base_cfg import Go2BaseRoughEnvCfg, Go2BaseRoughEnvCfg_PLAY


@configclass
class Go2TransformerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO Runner configuration with Causal Transformer policy (RoPE + Memory)."""
    
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "go2_transformer"
    empirical_normalization = False
    
    policy = RslRlPpoActorCriticCfg(
        # Use ActorCriticTransformer with Causal Attention
        class_name="ActorCriticTransformer",
        
        # Basic settings
        init_noise_std=1.0,
        activation="elu",
        
        # Embedding layer: Observations -> 256-dim token
        embedding_dim=256,
        embedding_hidden_dim=256,
        embedding_dropout=0.1,
        
        # Causal Transformer: 6 layers, 4 heads, FFN(256->512->256)
        # - RoPE: Rotary Position Embedding for better position encoding
        # - Memory: KV cache with window_size tokens of history
        transformer_layers=6,
        transformer_heads=4,
        transformer_feedforward_dim=512,
        transformer_dropout=0.1,
        window_size=128,     # Memory window size (number of past tokens to remember)
        rope_base=10000.0,   # RoPE base frequency (10000 is standard, higher = longer context)
        
        # Actor/Critic MLPs (process transformer output)
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# Define shared critic observations (same as policy, no privileged info)
@configclass
class SharedCriticObsCfg(ObsGroup):
    """Critic observations - same as policy (no privileged information)."""
    
    # Same observations as policy group
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.1, n_max=0.1))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)
    
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


# Environment configurations remain the same
@configclass
class Go2TransformerRoughEnvCfg(Go2BaseRoughEnvCfg):
    """Go2 environment with transformer policy.
    
    Note: Actor and Critic use the same observations (no privileged info for critic).
    This means the critic does NOT have access to:
    - base_lin_vel (linear velocity - not available in real deployment)
    - projected_gravity (can be noisy)
    - height_scan (terrain height map - requires sensor)
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Replace critic observations with shared observations (same as policy)
        self.observations.critic = SharedCriticObsCfg()


@configclass  
class Go2TransformerRoughEnvCfg_PLAY(Go2BaseRoughEnvCfg_PLAY):
    """Go2 play environment with transformer policy.
    
    Note: Actor and Critic use the same observations (no privileged info for critic).
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Apply same observation setup for play mode
        self.observations.critic = SharedCriticObsCfg()

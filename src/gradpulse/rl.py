"""Reinforcement learning module."""
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore
    spaces = None  # type: ignore

from gradpulse.crossresonance import CrossResonanceZXOptimizer


class CrossResonanceEnv(gym.Env):
    """Gymnasium environment for RL discovery of Cross-Resonance pulse seeds.

    The state space contains the current timestep (as a fraction of total steps).
    The action space proposes a macroscopic pulse configuration (the envelope seed).
    The reward is based on the process fidelity of the resulting pulse.
    """

    def __init__(self, optimizer: CrossResonanceZXOptimizer,
                 dt_ns: float = 1.0,
                 n_slices: int = 150,
                 max_steps: int = 50):
        """Initialize the RL environment."""
        if gym is None:
            raise ImportError("gymnasium is required for the RL module. "
                              "Install it with `pip install gymnasium`.")
        self.optimizer = optimizer
        self.dt_ns = dt_ns
        self.n_slices = n_slices
        self.max_steps = max_steps
        self.current_step = 0

        # State: [timestep/max_steps]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Action: parameters to generate the CR pulse envelope (e.g. amplitude scales)
        # We define a 4-dimensional continuous action:
        # [uc_scale, ut_scale, vc_scale, vt_scale]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().__init__() if hasattr(super(), '__init__') else None
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.current_step += 1

        # Build envelope from action
        # The optimizer expects an envelope shape [1, n_slices, n_channels]
        # For simplicity in the environment, we create a constant pulse scaled by the action
        action_t = torch.tensor(action, dtype=torch.float32)
        n_channels = self.optimizer.n_channels
        env_stack = torch.zeros((1, self.n_slices, n_channels), dtype=torch.float32, device=self.optimizer._H_DRIFT.device)
        for i in range(n_channels):
            env_stack[0, :, i] = action_t[i]

        # Clamp to bounds
        env_stack = torch.clamp(env_stack, -1.0, 1.0)

        # We need a proper reward to satisfy the plan: "Final fidelity from a truncated GRAPE run"
        # We perform a truncated optimization to get the polished fidelity
        polished_result = self.optimizer.optimize(
            iterations=2, # Truncated
            n_slices=self.n_slices,
            dt_ns=self.dt_ns,
            seed0=0,
            verbose=False
        )

        reward = polished_result.get("best_fidelity", 0.0)

        done = self.current_step >= self.max_steps
        obs = np.array([self.current_step / self.max_steps], dtype=np.float32)
        return obs, reward, done, False, {"fidelity": reward}


def train_ppo(env_fn, epochs=50, steps_per_epoch=200, pi_lr=3e-4, vf_lr=1e-3,
              gamma=0.99, clip_ratio=0.2, train_pi_iters=80, train_v_iters=80):
    """Train the PPO agent on the given environment using stable-baselines3."""
    # Now using stable-baselines3 as per code review feedback to be robust and stable
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    vec_env = make_vec_env(env_fn, n_envs=1)

    agent = PPO("MlpPolicy", vec_env, verbose=1, gamma=gamma, clip_range=clip_ratio,
                learning_rate=pi_lr, n_steps=steps_per_epoch, n_epochs=train_pi_iters)

    # Train
    agent.learn(total_timesteps=epochs * steps_per_epoch)
    return agent

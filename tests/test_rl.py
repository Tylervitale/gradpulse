import pytest
import numpy as np
import torch
try:
    import gymnasium as gym
except ImportError:
    gym = None

from gradpulse.rl import CrossResonanceEnv, train_ppo

class MockOptimizer:
    n_channels = 4
    def __init__(self):
        self._H_DRIFT = torch.zeros((10, 10))
    def optimize(self, iterations, n_slices, dt_ns, seed0, verbose):
        return {"best_fidelity": 0.99}

@pytest.fixture
def mock_optimizer():
    return MockOptimizer()

@pytest.mark.skipif(gym is None, reason="gymnasium not installed")
def test_cr_env_init(mock_optimizer):
    env = CrossResonanceEnv(mock_optimizer)
    assert env.observation_space.shape == (1,)
    assert env.action_space.shape == (4,)
    assert env.max_steps == 50

@pytest.mark.skipif(gym is None, reason="gymnasium not installed")
def test_cr_env_reset(mock_optimizer):
    env = CrossResonanceEnv(mock_optimizer)
    obs, info = env.reset()
    assert obs.shape == (1,)
    assert obs[0] == 0.0
    assert env.current_step == 0

@pytest.mark.skipif(gym is None, reason="gymnasium not installed")
def test_cr_env_step(mock_optimizer):
    env = CrossResonanceEnv(mock_optimizer, dt_ns=1.0, n_slices=10, max_steps=2)
    obs, info = env.reset()
    action = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

    # Take a step
    obs, reward, done, truncated, info = env.step(action)

    assert obs.shape == (1,)
    assert obs[0] == 0.5  # current_step / max_steps
    assert isinstance(reward, float)
    assert done is False
    assert truncated is False
    assert "fidelity" in info
    assert reward == 0.99

    # Take another step
    obs, reward, done, truncated, info = env.step(action)
    assert done is True
    assert obs[0] == 1.0

@pytest.mark.skipif(gym is None, reason="gymnasium not installed")
def test_train_ppo(mock_optimizer):
    def make_env():
        return CrossResonanceEnv(mock_optimizer, max_steps=5, n_slices=10)

    # Test short training loop
    agent = train_ppo(make_env, epochs=1, steps_per_epoch=10, pi_lr=1e-3, vf_lr=1e-3, train_pi_iters=1, train_v_iters=1)
    assert agent is not None

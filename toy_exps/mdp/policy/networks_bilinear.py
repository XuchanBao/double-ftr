import jax
from jax import random
import jax.numpy as np
import flax.linen as nn


def sample_action_gaussian(key, mean_action, log_std):
    key, subkey = random.split(key)
    action = random.normal(subkey, shape=mean_action.shape) * np.exp(log_std) + mean_action
    return key, np.array(action)


def init_weights(obj, key, inputs):
    key, subkey = random.split(key)
    weights = obj.init(subkey, inputs)
    return key, weights


def const_initializer(val, dtype=np.float_):
    def init(key, shape, dtype=dtype):
        return np.ones(shape, jax.dtypes.canonicalize_dtype(dtype)) * val

    return init


class Policy(nn.Module):
    action_dim: int
    init_log_std: float
    init_weight_mean: float

    @nn.compact
    def __call__(self, state):
        mean_action = self.param('action_mean', const_initializer(self.init_weight_mean), (self.action_dim,))

        action_log_stds = self.param('action_log_std', const_initializer(self.init_log_std), (self.action_dim,))

        return mean_action, action_log_stds


class PolicyFixedStd(nn.Module):
    action_dim: int
    init_log_std: float
    init_weight_mean: float

    @nn.compact
    def __call__(self, state):
        mean_action = self.param('action_mean', const_initializer(self.init_weight_mean), (self.action_dim,))

        return mean_action, np.ones_like(mean_action) * self.init_log_std
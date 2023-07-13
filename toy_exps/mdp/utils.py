from typing import NamedTuple

from jax import numpy as np
from typing import Union, Iterable, Mapping


class DataHistoryNamedTuple(NamedTuple):
    states: np.ndarray
    actions_u: np.ndarray
    actions_v: np.ndarray
    rewards_u: np.ndarray
    rewards_v: np.ndarray
    dones: np.ndarray


class ParameterNamedTuple(NamedTuple):
    critic_u: Union[np.ndarray, Iterable, Mapping, None]
    critic_v: Union[np.ndarray, Iterable, Mapping, None]
    policy_u: Union[np.ndarray, Iterable, Mapping]
    policy_v: Union[np.ndarray, Iterable, Mapping]

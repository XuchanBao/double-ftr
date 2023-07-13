import jax.numpy as np


def init_state(key):
    return key, np.array([0.])


def bilinear_game_step(state, action_u, action_v):
    reward_u = np.dot(action_u, action_v)
    reward_v = - reward_u
    done = True
    info = dict()
    return state, reward_u, reward_v, done, info

import jax
import jax.numpy as np


def get_advantage(reward_mat, value_mat, masks, gamma=0.99, tau=0.95):
    # pad 0's at the end along the last axis (the horizon axis)
    value_mat = np.pad(value_mat, ((0, 0), (0, 1)))
    gae = 0
    returns = []

    for step in reversed(range(reward_mat.shape[1])):       # reverse loop over trajectory length
        delta = reward_mat[:, step] + gamma * value_mat[:, step + 1] * masks[:, step] - value_mat[:, step]
        gae = delta + gamma * tau * masks[:, step] * gae
        returns.insert(0, gae + value_mat[:, step])
    returns = np.stack(returns).T   # (horizon, bs) --> (bs, horizon)
    return returns


def get_policy_obj_fn(policy, states, actions, advantages, log_action_prob_fn):
    def policy_obj_fn(weights):
        log_probs = log_action_prob_fn(policy.apply(weights, states), actions)  # shape = (# traj, horizon, action_dim)

        log_probs_traj_horizon = log_probs.sum(axis=-1, keepdims=True)     # shape = (# traj, horizon, 1)
        obj = - log_probs_traj_horizon * advantages     # shape = (# traj, horizon, 1)
        return obj.mean(axis=0).sum()

    return jax.jit(policy_obj_fn)


def get_critic_obj_fn(critic, states, returns):
    def critic_obj_fn(weights):
        vals = critic.apply(weights, states)
        loss = ((returns - vals) ** 2).mean()
        return loss
    return jax.jit(critic_obj_fn)


def log_action_prob_fn_gaussian(policy_outputs, actions):
    traj_means, traj_logstds = policy_outputs
    log_probs = jax.scipy.stats.norm.logpdf(actions, loc=traj_means, scale=np.exp(traj_logstds))
    return log_probs

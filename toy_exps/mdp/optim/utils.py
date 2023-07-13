from jax import grad, jacfwd, jacrev, jvp
from jax.flatten_util import ravel_pytree
import jax.numpy as np


def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]


def conjugate_gradient(mvp, b, damping=0.01, max_iter=30):
    x = np.zeros_like(b)
    r = mvp(b)
    p = r
    rdotr = r.dot(r)
    for i in range(1, max_iter):
        Ap = mvp(mvp(p)) + damping * p
        v = rdotr / p.dot(Ap)
        x += v * p
        r -= v * Ap
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
    return x


def hessian(f):
    return jacfwd(jacrev(f))


def get_trajectory_log_pi_adv_fn(policy, states, actions, advantages, unravel_fn, log_action_prob_fn):
    """
    Get function (flat weights) --> (log policy probability per trajectory)
    :param policy: policy object
    :param states: shape = (# of trajs, horizon, state_dim)
    :param actions: shape = (# of trajs, horizon, state_dim)
    :param advantages: shape = (# of trajs, horizon, 1)
    :param unravel_fn: function mapping (flattened weights) --> (pytree structure)
    :param log_action_prob_fn: function mapping (policy outputs, action) --> (log probability of action)
    :return:
    """

    def trajectory_log_pi_adv_fn(weights):
        """
        Policy probability * advantage per trajectory per timestep
        :param weights: flattened
        :return: policy probability multiplied by advantages, shape = (# of trajs, horizon)
        """
        log_probs = log_action_prob_fn(policy.apply(unravel_fn(weights), states), actions)

        # return (log_probs * advantages).sum(axis=tuple(range(1, log_probs.ndim)))      # sum all but the first axis
        return (log_probs * advantages).sum(axis=tuple(range(2, log_probs.ndim)))  # sum all but the first two axes

    return trajectory_log_pi_adv_fn


def get_trajectory_log_pi_fn(policy, states, actions, unravel_fn, log_action_prob_fn):
    """
    Get function (flat weights) --> (log policy probability per trajectory)
    :param policy: policy object
    :param states: shape = (# of trajs, horizon, state_dim)
    :param actions: shape = (# of trajs, horizon, state_dim)
    :param unravel_fn: function mapping (flattened weights) --> (pytree structure)
    :param log_action_prob_fn: function mapping (policy outputs, action) --> (log probability of action)
    :return:
    """

    def trajectory_log_pi_fn(weights):
        """
        Policy probability per trajectory per timestep
        :param weights: flattened
        :return: policy probability, shape = (# of trajs, horizon)
        """
        log_probs = log_action_prob_fn(policy.apply(unravel_fn(weights), states), actions)

        # return log_probs.sum(axis=tuple(range(1, log_probs.ndim)))      # sum all but the first two axes
        return log_probs.sum(axis=tuple(range(2, log_probs.ndim)))  # sum all but the first two axes

    return trajectory_log_pi_fn


def get_policy_gradient_and_jac(obj_fn_u, obj_fn_v, policy_u, policy_v, pu_weights, pv_weights,
                                states_per_traj, actions_u_per_traj, actions_v_per_traj, adv_u_per_traj,
                                log_action_prob_fn):
    pu_weight_flat, unravel_fn_u = ravel_pytree(pu_weights)
    pv_weight_flat, unravel_fn_v = ravel_pytree(pv_weights)

    flat_obj_fn_u = lambda w: obj_fn_u(unravel_fn_u(w))
    flat_obj_fn_v = lambda w: obj_fn_v(unravel_fn_v(w))

    # 1. get original gradients
    grad_u = grad(flat_obj_fn_u)(pu_weight_flat)
    grad_v = grad(flat_obj_fn_v)(pv_weight_flat)

    # 2. compute trajectory-wise statistics
    # trajectory-wise sum of log pi, shape = (n_trajs, horizon)
    traj_log_pi_fn_u = get_trajectory_log_pi_fn(policy_u, states_per_traj, actions_u_per_traj, unravel_fn_u,
                                                log_action_prob_fn=log_action_prob_fn)
    traj_log_pi_fn_v = get_trajectory_log_pi_fn(policy_v, states_per_traj, actions_v_per_traj, unravel_fn_v,
                                                log_action_prob_fn=log_action_prob_fn)

    # trajectory-wise sum of log pi * Adv, shape = (n_trajs, horizon)
    traj_log_pi_adv_fn_u = get_trajectory_log_pi_adv_fn(
        policy_u, states_per_traj, actions_u_per_traj, adv_u_per_traj, unravel_fn_u,
        log_action_prob_fn=log_action_prob_fn)
    traj_log_pi_adv_fn_v = get_trajectory_log_pi_adv_fn(
        policy_v, states_per_traj, actions_v_per_traj, adv_u_per_traj, unravel_fn_v,
        log_action_prob_fn=log_action_prob_fn)

    # 3. compute Jacobians shape = (n_traj, horizon, n_param)
    jac_logpi_u = jacfwd(traj_log_pi_fn_u)(pu_weight_flat)
    jac_logpi_adv_u = jacfwd(traj_log_pi_adv_fn_u)(pu_weight_flat)
    jac_logpi_v = jacfwd(traj_log_pi_fn_v)(pv_weight_flat)
    jac_logpi_adv_v = jacfwd(traj_log_pi_adv_fn_v)(pv_weight_flat)

    n_trajs = len(jac_logpi_adv_u)

    tensordot_axes = ([0, 1], [0, 1])
    huu = hessian(flat_obj_fn_u)(pu_weight_flat) + np.tensordot(jac_logpi_u, jac_logpi_adv_u,
                                                                axes=tensordot_axes) / float(n_trajs)
    huv = np.tensordot(jac_logpi_u, jac_logpi_adv_v, axes=tensordot_axes) / float(n_trajs)
    hvu = np.tensordot(jac_logpi_v, jac_logpi_adv_u, axes=tensordot_axes) / float(n_trajs)
    hvv = hessian(flat_obj_fn_v)(pv_weight_flat) + np.tensordot(jac_logpi_v, jac_logpi_adv_v,
                                                                axes=tensordot_axes) / float(n_trajs)

    return grad_u, grad_v, huu, huv, hvu, hvv

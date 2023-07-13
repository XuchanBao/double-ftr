import jax.numpy as np
from jax import grad, jacfwd
from jax.flatten_util import ravel_pytree
from mdp.optim.utils import hessian, get_trajectory_log_pi_fn, get_trajectory_log_pi_adv_fn


def get_lss_gradients(obj_fn_u, obj_fn_v, policy_u, policy_v, pu_weights, pv_weights,
                     states_per_traj,
                     actions_u_per_traj,
                     actions_v_per_traj,
                     adv_u_per_traj,
                     log_action_prob_fn):
    pu_ravelled, unravel_fn_u = ravel_pytree(pu_weights)
    pv_ravelled, unravel_fn_v = ravel_pytree(pv_weights)

    ravelled_obj_fn_u = lambda w: obj_fn_u(unravel_fn_u(w))
    ravelled_obj_fn_v = lambda w: obj_fn_v(unravel_fn_v(w))

    # 1. get original gradients
    grad_u = grad(ravelled_obj_fn_u)(pu_ravelled)
    grad_v = grad(ravelled_obj_fn_v)(pv_ravelled)

    # 2. compute trajectory-wise statistics
    # trajectory-wise sum of log pi, shape = (n_trajs, d)
    traj_log_pi_fn_u = get_trajectory_log_pi_fn(policy_u, states_per_traj, actions_u_per_traj, unravel_fn_u,
                                                log_action_prob_fn=log_action_prob_fn)
    traj_log_pi_fn_v = get_trajectory_log_pi_fn(policy_v, states_per_traj, actions_v_per_traj, unravel_fn_v,
                                                log_action_prob_fn=log_action_prob_fn)

    # trajectory-wise sum of log pi * Adv, shape = (n_trajs, d)
    traj_log_pi_adv_fn_u = get_trajectory_log_pi_adv_fn(
        policy_u, states_per_traj, actions_u_per_traj, adv_u_per_traj, unravel_fn_u,
        log_action_prob_fn=log_action_prob_fn)
    traj_log_pi_adv_fn_v = get_trajectory_log_pi_adv_fn(
        policy_v, states_per_traj, actions_v_per_traj, adv_u_per_traj, unravel_fn_v,
        log_action_prob_fn=log_action_prob_fn)

    # 3. compute hessian
    jac_logpi_u = jacfwd(traj_log_pi_fn_u)(pu_ravelled)
    jac_logpi_adv_u = jacfwd(traj_log_pi_adv_fn_u)(pu_ravelled)
    jac_logpi_v = jacfwd(traj_log_pi_fn_v)(pv_ravelled)
    jac_logpi_adv_v = jacfwd(traj_log_pi_adv_fn_v)(pv_ravelled)

    n_trajs = len(jac_logpi_adv_u)

    huu = hessian(ravelled_obj_fn_u)(pu_ravelled) + jac_logpi_u.T @ jac_logpi_adv_u / float(n_trajs)
    huv = jac_logpi_v.T @ jac_logpi_adv_u / float(n_trajs)
    hvu = jac_logpi_u.T @ jac_logpi_adv_v / float(n_trajs)
    hvv = hessian(ravelled_obj_fn_v)(pv_ravelled) + jac_logpi_v.T @ jac_logpi_adv_v / float(n_trajs)

    jac = np.vstack([np.hstack([huu, huv]), np.hstack([- hvu, -hvv])])
    grad_combine = np.concatenate([grad_u, grad_v])

    eps = 0.0001
    id_combine = np.eye(len(huu) + len(hvv))
    grad_second = jac.T @ np.linalg.solve(jac.T @ jac + eps * id_combine, jac.T @ grad_combine)

    total_grad_combine = 0.5 * grad_combine + 0.5 * grad_second

    total_grad_u = total_grad_combine[:len(grad_u)]
    total_grad_v = total_grad_combine[len(grad_u):]

    return unravel_fn_u(total_grad_u), unravel_fn_v(total_grad_v)

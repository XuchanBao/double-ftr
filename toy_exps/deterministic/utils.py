from typing import NamedTuple

import wandb
import jax.numpy as np
import numpy as onp
from jax import lax
from jax.flatten_util import ravel_pytree


class LQParams(NamedTuple):
    A: onp.ndarray
    Bu: onp.ndarray
    Bv: onp.ndarray
    Qu: onp.ndarray
    Qv: onp.ndarray
    Ru: onp.ndarray
    Rv: onp.ndarray


def get_default_lq_params(A, b, r, q):
    Bu = onp.array([[1.],
                    [1.]])
    Bv = onp.array([[b],
                    [1.]])
    Qu = onp.array([[0.01, 0.],
                    [0., 1.]])
    Qv = onp.array([[1., 0.],
                    [0., q]])
    Ru = onp.array([[0.01]])
    Rv = onp.array([[r]])

    return LQParams(A=A, Bu=Bu, Bv=Bv, Qu=Qu, Qv=Qv, Ru=Ru, Rv=Rv)


def fixed_point_solver(f, init_soln, tol=1e-6):
    def cond_fun(carry):
        z_prev, z = carry
        return np.linalg.norm(z_prev - z) > tol

    def body_fun(carry):
        _, z = carry
        return z, f(z)

    init_carry = (init_soln, f(init_soln))
    _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star


def wandb_log(lq_game, Ku, Kv, step, ravel_fn_ku, ravel_fn_kv):
    mean_rew_u, mean_rew_v = lq_game.evaluate(Ku, Kv, wandb.config.cov_rollout_steps)
    Ku_flat, _ = ravel_pytree(Ku)
    Kv_flat, _ = ravel_pytree(Kv)
    grad_u, grad_v = lq_game.compute_vector_field(Ku_flat=Ku_flat, Kv_flat=Kv_flat,
                                                  ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv,
                                                  cov_rollout_steps=wandb.config.cov_rollout_steps)
    j00, j01, j10, j11 = lq_game.compute_jacobian(Ku_flat=Ku_flat, Kv_flat=Kv_flat,
                                                  ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv,
                                                  cov_rollout_steps=wandb.config.cov_rollout_steps)
    jac = onp.vstack([onp.hstack([j00, j01]),
                      onp.hstack([j10, j11])])
    eig, eigv = onp.linalg.eig(jac)
    eig_re = onp.real(eig)
    eig_im = onp.imag(eig)

    eig00, _ = onp.linalg.eig(j00)
    eig00 = onp.real(eig00)
    eig11, _ = onp.linalg.eig(j11)
    eig11 = onp.real(eig11)

    wandb.log({
        'Weights/Ku_0': Ku[0, 0].item(),
        'Weights/Ku_1': Ku[0, 1].item(),
        'Weights/Kv_0': Kv[0, 0].item(),
        'Weights/Kv_1': Kv[0, 1].item(),
        'Grad/grad_u_norm': onp.linalg.norm(grad_u),
        'Grad/grad_v_norm': onp.linalg.norm(grad_v),
        'Jacobian/eig0_re': eig_re[0],
        'Jacobian/eig1_re': eig_re[1],
        'Jacobian/eig2_re': eig_re[2],
        'Jacobian/eig3_re': eig_re[3],
        'Jacobian/eig0_im': eig_im[0],
        'Jacobian/eig1_im': eig_im[1],
        'Jacobian/eig2_im': eig_im[2],
        'Jacobian/eig3_im': eig_im[3],
        'Hessian/fuu_eig0': eig00[0],
        'Hessian/fuu_eig1': eig00[1],
        'Hessian/gvv_eig0': eig11[0],
        'Hessian/gvv_eig1': eig11[1],
        'Loss/mean_reward_u': mean_rew_u,
        'Loss/mean_reward_v': mean_rew_v,
        'Episode': step
    }, step=step)

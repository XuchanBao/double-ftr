"""
Linear quadratic game (general-sum)
"""

import wandb
import time
import numpy as onp
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
from mdp.optim import AlgorithmTypes
from deterministic.lq_game_gs import GeneralSumLinearQuadraticGame
from deterministic.utils import wandb_log, get_default_lq_params

# import os
# os.environ['WANDB_MODE'] = 'dryrun'

default_params = dict(
    algo=AlgorithmTypes.FTR,
    n_episode=50000,
    lr=0.01,
    cov_rollout_steps=40,
    hinv_lambda=1e-4,
    fixed_point_tol=1e-5,
    use_precond_hgd=False,
    log_every=5
)

tags = []

run = wandb.init(project='exact-lq-general-sum', entity='follow-the-ridge', config=default_params, tags=tags)

A = onp.array([[0.511, 0.064],
               [0.533, 0.993]])
lq_params = get_default_lq_params(A=A, b=0., r=0.01, q=0.147)
lq_game = GeneralSumLinearQuadraticGame(lq_params, fixed_point_tol=wandb.config.fixed_point_tol)

Ku = onp.array([[0.5, 0.38]])
Kv = onp.array([[0.06, 0.60]])

Ku_flat, ravel_fn_ku = ravel_pytree(Ku)
Kv_flat, ravel_fn_kv = ravel_pytree(Kv)

wandb_log(lq_game, Ku, Kv, step=0, ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv)

for step_i in tqdm(range(wandb.config.n_episode)):
    t0 = time.time()

    if wandb.config.algo == AlgorithmTypes.GDA:
        grad_u, grad_v = lq_game.compute_vector_field(Ku_flat=Ku_flat, Kv_flat=Kv_flat,
                                                      ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv,
                                                      cov_rollout_steps=wandb.config.cov_rollout_steps)

    elif wandb.config.algo == AlgorithmTypes.FTR:
        vec_u, vec_v = lq_game.compute_vector_field(Ku_flat=Ku_flat, Kv_flat=Kv_flat,
                                                    ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv,
                                                    cov_rollout_steps=wandb.config.cov_rollout_steps)
        fuu, fuv, gvu, gvv = lq_game.compute_jacobian(Ku_flat=Ku_flat, Kv_flat=Kv_flat,
                                                      ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv,
                                                      cov_rollout_steps=wandb.config.cov_rollout_steps)

        id_u = np.eye(len(fuu))
        id_v = np.eye(len(gvv))

        # The two implementations shouldn't make a big difference
        if wandb.config.use_precond_hgd:
            # preconditioned Hamiltonian GD (damping added to both vec and correction term)
            h_grad_u, h_grad_v = lq_game.hamiltonian_grad(Ku_flat=Ku_flat, Kv_flat=Kv_flat,
                                                          ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv,
                                                          cov_rollout_steps=wandb.config.cov_rollout_steps)

            grad_u = np.linalg.solve(fuu.T @ fuu + wandb.config.hinv_lambda * id_u, fuu @ h_grad_u)
            grad_v = np.linalg.solve(gvv.T @ gvv + wandb.config.hinv_lambda * id_v, gvv @ h_grad_v)
        else:
            # original implementation
            corr_u = np.linalg.solve(fuu @ fuu.T + wandb.config.hinv_lambda * id_u, fuu @ gvu.T @ vec_v)
            corr_v = np.linalg.solve(gvv @ gvv.T + wandb.config.hinv_lambda * id_v, gvv @ fuv.T @ vec_u)
            grad_u = vec_u + corr_u
            grad_v = vec_v + corr_v

    elif wandb.config.algo == AlgorithmTypes.HGD:
        grad_u, grad_v = lq_game.hamiltonian_grad(Ku_flat=Ku_flat, Kv_flat=Kv_flat,
                                                  ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv,
                                                  cov_rollout_steps=wandb.config.cov_rollout_steps)
    else:
        raise NotImplementedError

    if step_i < 2:
        print(f"step = {step_i + 1}, took {time.time() - t0} s")

    Ku_flat -= wandb.config.lr * grad_u
    Kv_flat -= wandb.config.lr * grad_v

    Ku = ravel_fn_ku(Ku_flat)
    Kv = ravel_fn_kv(Kv_flat)

    if (step_i + 1) % wandb.config.log_every == 0:
        wandb_log(lq_game, Ku, Kv, step=step_i + 1, ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv)

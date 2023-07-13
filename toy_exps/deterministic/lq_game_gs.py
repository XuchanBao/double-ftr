"""
Deterministic general-sum linear quadratic game
"""
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax import jit, jacfwd
from functools import partial
import numpy as onp
from deterministic.utils import fixed_point_solver, LQParams


class GeneralSumLinearQuadraticGame:
    def __init__(self, lq_params: LQParams, fixed_point_tol=1e-6):
        self.A = lq_params.A
        self.Bu = lq_params.Bu
        self.Bv = lq_params.Bv
        self.Qu = lq_params.Qu
        self.Qv = lq_params.Qv
        self.Ru = lq_params.Ru
        self.Rv = lq_params.Rv

        self.state_dim = self.A.shape[0]
        self.action_dim_u = self.Bu.shape[1]
        self.action_dim_v = self.Bv.shape[1]

        self.fixed_point_tol = fixed_point_tol

    @partial(jit, static_argnums=(0, 1, 2, 3))
    def compute_vector_field(self, cov_rollout_steps, ravel_fn_ku, ravel_fn_kv, Ku_flat, Kv_flat):
        # Compute Pu, Pv by solving the Lyapunov equations
        Ku = ravel_fn_ku(Ku_flat)
        Kv = ravel_fn_kv(Kv_flat)

        Pu, Pv = self._solve_lyapunov_eqns(Ku, Kv)

        # Estimate the state covariance
        state_cov = self._compute_state_cov(Ku, Kv, cov_rollout_steps)

        # vector field
        A_bar = self.A - self.Bu @ Ku - self.Bv @ Kv
        w_u = 2. * (self.Ru @ Ku - self.Bu.T @ Pu @ A_bar) @ state_cov
        w_v = 2. * (self.Rv @ Kv - self.Bv.T @ Pv @ A_bar) @ state_cov

        w_u_flat, ravel_fn_wu = ravel_pytree(w_u)
        w_v_flat, ravel_fn_wv = ravel_pytree(w_v)

        return w_u_flat, w_v_flat

    # TODO: this method currently unused. Make it a configure option
    @partial(jit, static_argnums=(0, 1, 2, 3))
    def compute_vector_field_finite_length(self, rollout_steps, ravel_fn_ku, ravel_fn_kv, Ku_flat, Kv_flat):
        w_u = - jacfwd(lambda k: self.evaluate(ravel_fn_ku(k), ravel_fn_kv(Kv_flat), rollout_steps))(Ku_flat)[0]
        w_v = - jacfwd(lambda k: self.evaluate(ravel_fn_ku(Ku_flat), ravel_fn_kv(k), rollout_steps))(Kv_flat)[1]
        w_u_flat, ravel_fn_wu = ravel_pytree(w_u)
        w_v_flat, ravel_fn_wv = ravel_pytree(w_v)

        return w_u_flat, w_v_flat

    @partial(jit, static_argnums=(0, 1, 2, 3))
    def compute_hamiltonian(self, cov_rollout_steps, ravel_fn_ku, ravel_fn_kv, Ku_flat, Kv_flat):
        w_u_flat, w_v_flat = self.compute_vector_field(cov_rollout_steps, ravel_fn_ku, ravel_fn_kv, Ku_flat, Kv_flat)
        hamiltonian = 0.5 * (np.linalg.norm(w_u_flat) ** 2 + np.linalg.norm(w_v_flat) ** 2)
        return hamiltonian

    @partial(jit, static_argnums=(0, 1, 2, 3))
    def compute_jacobian(self, cov_rollout_steps, ravel_fn_ku, ravel_fn_kv, Ku_flat, Kv_flat):
        jac0_, jac1_ = jacfwd(
            lambda k: self.compute_vector_field(cov_rollout_steps, ravel_fn_ku, ravel_fn_kv, *k))((Ku_flat, Kv_flat))
        return jac0_[0], jac0_[1], jac1_[0], jac1_[1]

    @partial(jit, static_argnums=(0, 1, 2, 3))
    def hamiltonian_grad(self, cov_rollout_steps, ravel_fn_ku, ravel_fn_kv, Ku_flat, Kv_flat):
        h_grad = jacfwd(
            lambda k: self.compute_hamiltonian(cov_rollout_steps, ravel_fn_ku, ravel_fn_kv, *k)
        )((Ku_flat, Kv_flat))

        return h_grad[0], h_grad[1]

    @partial(jit, static_argnums=(0, 3))
    def evaluate(self, Ku, Kv, rollout_steps):
        init_states, init_probs = self._get_init_state_dist()
        mean_reward_u = 0.
        mean_reward_v = 0.
        for z0, prob in zip(init_states, init_probs):
            _, rew_u, rew_v = self._rollout(num_steps=rollout_steps, Ku=Ku, Kv=Kv, z0=z0)
            mean_reward_u += prob * rew_u
            mean_reward_v += prob * rew_v
        return mean_reward_u, mean_reward_v

    def lyapunov_iteration(self, tol=1e-5):
        def approx_k(pu, pv):
            ku_new = onp.linalg.solve(self.Ru + self.Bu.T @ pu @ self.Bu, self.Bu.T @ pu @ self.A)
            kv_new = onp.linalg.solve(self.Rv + self.Bv.T @ pv @ self.Bv, self.Bv.T @ pv @ self.A)
            ku = onp.zeros_like(ku_new)
            kv = onp.zeros_like(kv_new)

            k_step = 1
            while k_step < 5000 and onp.linalg.norm(ku_new - ku) + onp.linalg.norm(kv_new - kv) > tol:
                ku, kv = ku_new, kv_new
                ku_new = onp.linalg.solve(self.Ru + self.Bu.T @ pu @ self.Bu, self.Bu.T @ pu @ (self.A - self.Bv @ kv))
                kv_new = onp.linalg.solve(self.Rv + self.Bv.T @ pv @ self.Bv, self.Bv.T @ pv @ (self.A - self.Bu @ ku))
                k_step += 1

            return ku_new, kv_new

        def iteration_step(pu, pv, ku, kv):
            a_bar = self.A - self.Bu @ ku - self.Bv @ kv

            pu = a_bar.T @ pu @ a_bar + ku.T @ self.Ru @ ku + self.Qu
            pv = a_bar.T @ pv @ a_bar + kv.T @ self.Rv @ kv + self.Qv

            return pu, pv

        Pu, Pv = onp.eye(self.state_dim), onp.eye(self.state_dim)
        Pu_new, Pv_new = iteration_step(Pu, Pv, *approx_k(Pu, Pv))

        total_steps = 1
        while total_steps < 1000 and onp.linalg.norm(Pu_new - Pu) + onp.linalg.norm(Pv_new - Pv) > tol:
            Pu, Pv = Pu_new, Pv_new
            Pu_new, Pv_new = iteration_step(Pu, Pv, *approx_k(Pu, Pv))
            total_steps += 1

        return approx_k(Pu_new, Pv_new)

    # ---- Helper functions ----

    @staticmethod
    def _get_init_state_dist():
        states = [onp.array([[1.], [1.]]), onp.array([[1.], [1.1]])]
        probs = [0.5, 0.5]
        return states, probs

    def step(self, state, Ku_z, Kv_z):
        a_u = - Ku_z
        a_v = - Kv_z

        next_state = self.A @ state + self.Bu @ a_u + self.Bv @ a_v
        cost_u = state.T @ self.Qu @ state + a_u.T @ self.Ru @ a_u
        cost_v = state.T @ self.Qv @ state + a_v.T @ self.Rv @ a_v

        reward_u = - cost_u
        reward_v = - cost_v

        done = False
        info = dict()
        return next_state, reward_u, reward_v, done, info

    @partial(jit, static_argnums=(0, 1))
    def _rollout(self, num_steps, Ku, Kv, z0):
        state, total_rew_u, total_rew_v, _, __ = self.step(z0, Ku @ z0, Kv @ z0)
        state_list = [z0, state]

        for step_i in range(1, num_steps):
            state, rew_u, rew_v, _, __ = self.step(state, Ku @ state, Kv @ state)
            total_rew_u += rew_u
            total_rew_v += rew_v
            state_list.append(state)
        return state_list, total_rew_u, total_rew_v

    def _compute_state_cov(self, Ku, Kv, rollout_steps):
        init_states, init_probs = self._get_init_state_dist()
        state_covs = []
        for z0 in init_states:
            z_list, _, __ = self._rollout(num_steps=rollout_steps, Ku=Ku, Kv=Kv, z0=z0)
            state_covs.append(np.array(list(map(lambda z: z @ z.T, z_list))).sum(axis=0))

        state_cov = np.array(list(map(lambda cov, p: p * cov, state_covs, init_probs))).sum(axis=0)
        return state_cov

    def _solve_lyapunov_eqns(self, Ku, Kv):
        def lyapunov_fn_Pu(pu):
            return A_bar.T @ pu @ A_bar + Ku.T @ self.Ru @ Ku + self.Qu

        def lyapunov_fn_Pv(pv):
            return A_bar.T @ pv @ A_bar + Kv.T @ self.Rv @ Kv + self.Qv

        A_bar = self.A - self.Bu @ Ku - self.Bv @ Kv

        Pu = fixed_point_solver(lyapunov_fn_Pu, init_soln=onp.eye(self.state_dim), tol=self.fixed_point_tol)
        Pv = fixed_point_solver(lyapunov_fn_Pv, init_soln=onp.eye(self.state_dim), tol=self.fixed_point_tol)

        return Pu, Pv

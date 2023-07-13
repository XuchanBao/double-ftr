import jax.numpy as np
from jax.flatten_util import ravel_pytree


def get_fr_gradients(pu_weights, pv_weights, grad_u, grad_v, huu, huv, hvu, hvv, hinv_lambda, ):
    pu_ravelled, unravel_fn_u = ravel_pytree(pu_weights)
    pv_ravelled, unravel_fn_v = ravel_pytree(pv_weights)

    id_u = np.eye(len(huu))
    id_v = np.eye(len(hvv))

    lamb_u = hinv_lambda
    lamb_v = hinv_lambda

    corr_u = np.linalg.solve(huu @ huu + lamb_u * id_u, huu @ huv @ grad_v)
    corr_v = np.linalg.solve(hvv @ hvv + lamb_v * id_v, hvv @ hvu @ grad_u)

    eps_damping = 1e-4
    damping = np.exp(- eps_damping * (np.dot(corr_u, corr_u) + np.dot(corr_v, corr_v)))
    total_grad_u = grad_u - damping * corr_u
    total_grad_v = grad_v - damping * corr_v

    info = {
        'lambda u': lamb_u, 'lambda v': lamb_v,
        'grad-norm/grad-u': np.linalg.norm(grad_u),
        'grad-norm/grad-v': np.linalg.norm(grad_v),
        'grad-norm/correction-u': np.linalg.norm(corr_u),
        'grad-norm/correction-v': np.linalg.norm(corr_v),
        'grad-norm/huu-cond': np.linalg.cond(huu),
        'grad-norm/huu-sq-cond': np.linalg.cond(huu @ huu),
        'grad-norm/huu-sq-lambda-cond': np.linalg.cond(huu @ huu + lamb_u * id_u),
        'grad-norm/hvv-cond': np.linalg.cond(hvv),
        'grad-norm/hvv-sq-cond': np.linalg.cond(hvv @ hvv),
        'grad-norm/hvv-sq-lambda-cond': np.linalg.cond(hvv @ hvv + lamb_v * id_v),
        'damping': damping
    }
    return unravel_fn_u(total_grad_u), unravel_fn_v(total_grad_v), info

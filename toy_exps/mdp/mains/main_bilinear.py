import wandb
import jax
from jax import random
import optax

from mdp.envs.bilinear_game import bilinear_game_step, init_state
from mdp.policy.networks_bilinear import Policy, PolicyFixedStd, init_weights, sample_action_gaussian
from mdp.policy.utils import log_action_prob_fn_gaussian
from mdp.train import get_simultate_episode_fn, get_update_weights_fn, train
from mdp.optim import AlgorithmTypes
from mdp.utils import ParameterNamedTuple

# import os
# os.environ['WANDB_MODE'] = 'dryrun'

default_params = dict(
    algo=AlgorithmTypes.FTR,
    n_episode=400,
    batch_size=2000,
    jit_batch_size=10,
    horizon=1,
    lr_actor=0.01,
    init_logstd=0.2,
    hinv_lambda=1e-4,
    update_std=True,
    seed=1234
)

tags = [
    'fix-hessian',
    'damping',
    'refactor-05-06'
]
run = wandb.init(project='bilinear', entity='follow-the-ridge', config=default_params, tags=tags)

key = random.PRNGKey(wandb.config.seed)

key, state = init_state(key)

policy_class = Policy if wandb.config.update_std else PolicyFixedStd
policy_u = policy_class(action_dim=1, init_log_std=wandb.config.init_logstd, init_weight_mean=1.)
key, pu_weights = init_weights(policy_u, key, state)

policy_v = policy_class(action_dim=1, init_log_std=wandb.config.init_logstd, init_weight_mean=1.)
key, pv_weights = init_weights(policy_v, key, state)

optim_u = optax.sgd(learning_rate=wandb.config.lr_actor)
optim_u_state = optim_u.init(pu_weights)
optim_v = optax.sgd(learning_rate=wandb.config.lr_actor)
optim_v_state = optim_v.init(pv_weights)

# record policy weights
p_u_mean = pu_weights['params']['action_mean'].item()
p_v_mean = pv_weights['params']['action_mean'].item()
if wandb.config.update_std:
    p_u_std = pu_weights['params']['action_log_std'].item()
    p_v_std = pv_weights['params']['action_log_std'].item()
else:
    p_u_std = wandb.config.init_logstd
    p_v_std = wandb.config.init_logstd

simulate_episode_fn = jax.jit(get_simultate_episode_fn(policy_u, policy_v, init_state, bilinear_game_step, wandb.config,
                                                       sample_action_fn=sample_action_gaussian))

update_weights_fn = jax.jit(get_update_weights_fn(
    config=wandb.config,
    policy_u=policy_u, policy_v=policy_v, critic=None,
    optim_u=optim_u, optim_v=optim_v, optim_critic_u=None, optim_critic_v=None,
    traj_shape=(wandb.config.batch_size, wandb.config.horizon),
    log_action_prob_fn=log_action_prob_fn_gaussian))

total_step = 0

wandb.log({
    'Weights/u': p_u_mean,
    'Weights/u_log_std': p_u_std,
    'Weights/v': p_v_mean,
    'Weights/v_log_std': p_v_std
}, step=total_step)

weights = ParameterNamedTuple(critic_u=None, critic_v=None, policy_u=pu_weights, policy_v=pv_weights)
optim_states = ParameterNamedTuple(critic_u=None, critic_v=None, policy_u=optim_u_state, policy_v=optim_v_state)

train(key, simulate_episode_fn, update_weights_fn,
      weights, optim_states, wandb.config)

import wandb
import jax
import optax
import time
from tqdm import tqdm

from mdp.policy.utils import get_advantage, get_policy_obj_fn, get_critic_obj_fn
from mdp.data.data_history import DataHistory
from mdp.utils import DataHistoryNamedTuple, ParameterNamedTuple
from mdp.optim import AlgorithmTypes
from mdp.optim.ftr import get_fr_gradients
from mdp.optim.utils import get_policy_gradient_and_jac
from mdp.optim.lss import get_lss_gradients


def get_simultate_episode_fn(policy_u, policy_v, init_state, game_step, config, sample_action_fn):
    def simulate_episode(key, pu_weights, pv_weights):
        traj_history = DataHistory()

        for sub_batch_i in range(config.jit_batch_size):
            key, state = init_state(key)

            for step_i in range(config.horizon):
                policy_out_u = policy_u.apply(pu_weights, state)
                key, action_u = sample_action_fn(key, *policy_out_u)

                policy_out_v = policy_v.apply(pv_weights, state)
                key, action_v = sample_action_fn(key, *policy_out_v)

                new_state, reward_u, reward_v, done, _ = game_step(state, action_u, action_v)

                if step_i + 1 == config.horizon:
                    done = True
                traj_history.add(state, action_u, action_v, reward_u, reward_v, 1 - done)

                state = new_state
        return key, traj_history.get_history_dict()

    return simulate_episode


def get_update_weights_fn(config, policy_u, policy_v, critic, optim_u, optim_v, optim_critic_u, optim_critic_v,
                          traj_shape, log_action_prob_fn):
    def update_weights(weights: ParameterNamedTuple,
                       optim_states: ParameterNamedTuple,
                       data: DataHistoryNamedTuple):
        # Calculate returns and advantages
        if critic is not None:
            val_u = critic.apply(weights.critic_u, data.states)
            val_v = critic.apply(weights.critic_v, data.states)

            returns_u = get_advantage(reward_mat=data.rewards_u.reshape(traj_shape),
                                      value_mat=val_u.reshape(traj_shape),
                                      masks=data.dones.reshape(traj_shape))  # (bs, horizon)
            returns_u = returns_u.reshape(val_u.shape)  # (bs * horizon, 1)
            adv_u = returns_u - val_u

            returns_v = get_advantage(reward_mat=data.rewards_v.reshape(traj_shape),
                                      value_mat=val_v.reshape(traj_shape),
                                      masks=data.dones.reshape(traj_shape))
            returns_v = returns_v.reshape(val_v.shape)
            adv_v = returns_v - val_v

            # Update critic
            cu_obj_fn = get_critic_obj_fn(critic, data.states, returns_u)
            cu_loss, cu_grad = jax.value_and_grad(cu_obj_fn)(weights.critic_u)
            cu_update, optim_cu_state = optim_critic_u.update(cu_grad, optim_states.critic_u)

            cv_obj_fn = get_critic_obj_fn(critic, data.states, returns_v)
            cv_loss, cv_grad = jax.value_and_grad(cv_obj_fn)(weights.critic_v)
            cv_update, optim_cv_state = optim_critic_v.update(cv_grad, optim_states.critic_v)
        else:
            adv_u = data.rewards_u[:, None]  # (bs, 1)
            adv_v = data.rewards_v[:, None]
            cu_loss, cv_loss = None, None
            cu_update, cv_update = None, None
            optim_cu_state, optim_cv_state = None, None

        # Update actors u and v
        obj_fn_u = get_policy_obj_fn(policy_u,
                                     states=data.states, actions=data.actions_u, advantages=adv_u,
                                     log_action_prob_fn=log_action_prob_fn)
        obj_fn_v = get_policy_obj_fn(policy_v,
                                     states=data.states, actions=data.actions_v, advantages=adv_v,
                                     log_action_prob_fn=log_action_prob_fn)

        log_dict = dict()
        if config.algo == AlgorithmTypes.GDA:
            loss_u, grad_u = jax.value_and_grad(obj_fn_u)(weights.policy_u)
            loss_v, grad_v = jax.value_and_grad(obj_fn_v)(weights.policy_v)

        elif config.algo == AlgorithmTypes.FTR:
            grad_u, grad_v, log_dict = get_fr_gradients(
                weights.policy_u, weights.policy_v,
                *get_policy_gradient_and_jac(obj_fn_u, obj_fn_v, policy_u, policy_v, weights.policy_u, weights.policy_v,
                                             data.states.reshape((*traj_shape, -1)),
                                             data.actions_u.reshape((*traj_shape, -1)),
                                             data.actions_v.reshape((*traj_shape, -1)),
                                             adv_u.reshape((*traj_shape, -1)),
                                             log_action_prob_fn=log_action_prob_fn),
                hinv_lambda=config.hinv_lambda
            )
            loss_u = obj_fn_u(weights.policy_u)
            loss_v = obj_fn_v(weights.policy_v)

        elif config.algo == AlgorithmTypes.LSS:
            grad_u, grad_v = get_lss_gradients(obj_fn_u, obj_fn_v, policy_u, policy_v, weights.policy_u,
                                               weights.policy_v,
                                               data.states.reshape((*traj_shape, -1)),
                                               data.actions_u.reshape((*traj_shape, -1)),
                                               data.actions_v.reshape((*traj_shape, -1)),
                                               adv_u.reshape((*traj_shape, -1)),
                                               log_action_prob_fn=log_action_prob_fn)
            loss_u = obj_fn_u(weights.policy_u)
            loss_v = obj_fn_v(weights.policy_v)

        else:
            raise NotImplementedError(f"Update algorithm {config.algo} is not implemented!")

        pu_update, optim_pu_state = optim_u.update(grad_u, optim_states.policy_u)
        pv_update, optim_pv_state = optim_v.update(grad_v, optim_states.policy_v)

        new_optim_state = ParameterNamedTuple(critic_u=optim_cu_state, critic_v=optim_cv_state,
                                              policy_u=optim_pu_state, policy_v=optim_pv_state)

        new_weights = ParameterNamedTuple(
            critic_u=optax.apply_updates(weights.critic_u, cu_update),
            critic_v=optax.apply_updates(weights.critic_v, cv_update),
            policy_u=optax.apply_updates(weights.policy_u, pu_update),
            policy_v=optax.apply_updates(weights.policy_v, pv_update)
        )

        log_dict.update({
            'Loss/mean_obj_u': loss_u,
            'Loss/mean_obj_v': loss_v,
            "Loss/critic_u": cu_loss,
            "Loss/critic_v": cv_loss,
            'Advantage/u': adv_u.mean(),
            'Advantage/v': adv_v.mean(),
        })

        return new_weights, new_optim_state, log_dict

    return update_weights


def train(key, simulate_episode_fn, update_weights_fn,
          weights: ParameterNamedTuple, optim_states: ParameterNamedTuple, config):
    total_step = 0
    for eps_i in tqdm(range(config.n_episode)):
        t0 = time.time()

        history = DataHistory()

        for batch_i in range(config.batch_size // config.jit_batch_size):
            key, traj_dict = simulate_episode_fn(key, weights.policy_u, weights.policy_v)

            history.add_dict(traj_dict)

            total_step += config.horizon * config.jit_batch_size

        if eps_i < 2:
            print(f"Simulation took {time.time() - t0} s")

        t0 = time.time()
        history_tuple = history.get_history_tuple()
        weights, optim_states, log_dict = update_weights_fn(
            weights,
            optim_states,
            history_tuple
        )
        if eps_i < 2:
            print(f"Update took {time.time() - t0} s")

        # record policy weights
        if 'action_mean' in weights.policy_u['params'].keys():
            if wandb.config.update_std:
                p_u_std = weights.policy_u['params']['action_log_std'].item()
                p_v_std = weights.policy_v['params']['action_log_std'].item()
            else:
                p_u_std = wandb.config.init_logstd
                p_v_std = wandb.config.init_logstd

            p_u_mean = weights.policy_u['params']['action_mean'].item()
            p_v_mean = weights.policy_v['params']['action_mean'].item()
            log_dict_weights = {
                'Weights/u': p_u_mean,
                'Weights/u_log_std': p_u_std,
                'Weights/v': p_v_mean,
                'Weights/v_log_std': p_v_std,
            }

        elif 'unnormalized_probs' in weights.policy_u['params'].keys():
            p_u1 = weights.policy_u['params']['unnormalized_probs'][0].item()
            p_u2 = weights.policy_u['params']['unnormalized_probs'][1].item()
            p_v1 = weights.policy_v['params']['unnormalized_probs'][0].item()
            p_v2 = weights.policy_v['params']['unnormalized_probs'][1].item()
            log_dict_weights = {
                'Weights/u_1': p_u1,
                'Weights/u_2': p_u2,
                'Weights/v_1': p_v1,
                'Weights/v_2': p_v2,
                'Weights/sm_u': jax.nn.softmax(weights.policy_u['params']['unnormalized_probs'])[0].item(),
                'Weights/sm_v': jax.nn.softmax(weights.policy_v['params']['unnormalized_probs'])[0].item(),
            }
        else:
            if wandb.config.update_std:
                p_u_std = weights.policy_u['params']['action_log_std'].item()
                p_v_std = weights.policy_v['params']['action_log_std'].item()
            else:
                p_u_std = wandb.config.init_logstd
                p_v_std = wandb.config.init_logstd

            log_dict_weights = dict()
            pu_weights_kernel = weights.policy_u['params']['weight']['kernel']
            pv_weights_kernel = weights.policy_v['params']['weight']['kernel']
            for weight_i in range(pu_weights_kernel.shape[0]):
                for weight_j in range(pu_weights_kernel.shape[1]):
                    log_dict_weights[f'Weights/u_{weight_i}{weight_j}'] = pu_weights_kernel[weight_i, weight_j].item()
                    log_dict_weights[f'Weights/v_{weight_i}{weight_j}'] = pv_weights_kernel[weight_i, weight_j].item()
            log_dict_weights['Weights/u_log_std'] = p_u_std
            log_dict_weights['Weights/v_log_std'] = p_v_std

        reward_u_per_traj = history_tuple.rewards_u.reshape((-1, config.horizon)).sum(-1)
        reward_v_per_traj = history_tuple.rewards_v.reshape((-1, config.horizon)).sum(-1)
        wandb.log({
            'Episode': eps_i,
            'Loss/reward_u_mean': reward_u_per_traj.mean(),
            'Loss/reward_v_mean': reward_v_per_traj.mean(),
            **log_dict_weights,
            **log_dict
        }, step=total_step)

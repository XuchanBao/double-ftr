import numpy as onp
import wandb

from deterministic.lq_game_gs import GeneralSumLinearQuadraticGame


def other(player_or_axis):
    if player_or_axis == 'u':
        return 'v'
    if player_or_axis == 'v':
        return 'u'
    if player_or_axis == 'x':
        return 'y'
    if player_or_axis == 'y':
        return 'x'
    raise ValueError("Argument should be one of ['u', 'v', 'x', 'y']!")


def get_centers_from_trajs(trajs):
    centers = {}
    for player in trajs.keys():
        centers[player] = {}
        for axis in trajs[player].keys():
            centers[player][axis] = onp.mean(trajs[player][axis])
    return centers


def get_paddings_from_trajs(centers, trajs, multiplier=2.):
    paddings = {}
    for player in trajs.keys():
        paddings[player] = {}
        for axis in trajs[player].keys():
            traj_max = max(trajs[player][axis])
            traj_min = min(trajs[player][axis])
            paddings[player][axis] = max(
                traj_max - centers[player][axis], centers[player][axis] - traj_min
            ) * multiplier
    return paddings


def get_meshes_for_slice(centers: dict, curr_k: dict, slice2d: dict, paddings: dict, n_points_per_side: int):
    mesh_edges = {}

    meshes = {
        'u': {},
        'v': {}
    }

    for axis in ['x', 'y']:
        axis_player = slice2d[axis]

        mesh_edges[axis] = onp.linspace(
            centers[axis_player][axis] - paddings[axis_player][axis],
            centers[axis_player][axis] + paddings[axis_player][axis],
            n_points_per_side
        )

    x_player = slice2d['x']
    y_player = slice2d['y']

    meshes[x_player]['x'], meshes[y_player]['y'] = onp.meshgrid(mesh_edges['x'], mesh_edges['y'])

    for axis in ['x', 'y']:
        axis_player = slice2d[axis]
        meshes[other(axis_player)][axis] = curr_k[other(axis_player)][axis] * onp.ones_like(meshes[x_player]['x'])

    return meshes


def get_traj_for_slice(trajs: dict, slice2d: dict):
    traj = {}
    for axis in ['x', 'y']:
        axis_player = slice2d[axis]
        traj[axis] = trajs[axis_player][axis]
    return traj


def get_quad_approx_reward_for_slice(jac, k_meshes, n_points_per_side, centers=None):
    reward_mesh = onp.zeros_like(k_meshes['u']['x'])

    if centers is not None:
        center_flat = onp.array([centers['u']['x'], centers['u']['y'], centers['v']['x'], centers['v']['y']])
    else:
        center_flat = None

    for x_i in range(n_points_per_side):
        for y_i in range(n_points_per_side):
            ku_flat = onp.array([k_meshes['u']['x'][x_i][y_i], k_meshes['u']['y'][x_i][y_i]])
            kv_flat = onp.array([k_meshes['v']['x'][x_i][y_i], k_meshes['v']['y'][x_i][y_i]])
            k_flat = onp.concatenate([ku_flat, kv_flat])
            if centers is not None:
                k_flat = k_flat - center_flat  # center the current vector

            r = onp.dot(k_flat, jac @ k_flat)
            reward_mesh[x_i][y_i] = r

    return reward_mesh


def get_rewards_mesh_for_slice(game: GeneralSumLinearQuadraticGame, k_meshes, n_points_per_side):
    reward_meshes = {
        'u': onp.zeros_like(k_meshes['u']['x']),
        'v': onp.zeros_like(k_meshes['u']['x']),
    }
    for x_i in range(n_points_per_side):
        for y_i in range(n_points_per_side):
            ku = onp.array([[k_meshes['u']['x'][x_i][y_i], k_meshes['u']['y'][x_i][y_i]]])
            kv = onp.array([[k_meshes['v']['x'][x_i][y_i], k_meshes['v']['y'][x_i][y_i]]])

            r_u, r_v = game.evaluate(ku, kv, rollout_steps=30)
            reward_meshes['u'][x_i][y_i] = r_u
            reward_meshes['v'][x_i][y_i] = r_v
    return reward_meshes


def get_trajs_from_wandb(filter_tags):
    api = wandb.Api()
    runs = list(api.runs("follow-the-ridge/exact-lq-general-sum", filters={'tags': {'$in': filter_tags}}))
    max_step = 10000

    data_all = []
    for run in runs:
        hist_list = list(
            run.scan_history(keys=['Episode', 'Weights/Ku_0', 'Weights/Ku_1', 'Weights/Kv_0', 'Weights/Kv_1'],
                             max_step=max_step))

        data = {'Episode': [], 'Weights/Ku_0': [], 'Weights/Ku_1': [], 'Weights/Kv_0': [], 'Weights/Kv_1': []}
        for k in data.keys():
            data[k] = [hist_element[k] for hist_element in hist_list]
        data_all.append({
            'u': {'x': data['Weights/Ku_0'], 'y': data['Weights/Ku_1']},
            'v': {'x': data['Weights/Kv_0'], 'y': data['Weights/Kv_1']}
        })

    return data_all, runs

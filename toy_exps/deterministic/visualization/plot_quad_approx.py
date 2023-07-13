import matplotlib
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
import numpy as onp

from deterministic.utils import get_default_lq_params
from deterministic.lq_game_gs import GeneralSumLinearQuadraticGame
from deterministic.visualization.plot_utils import get_paddings_from_trajs, get_meshes_for_slice, get_traj_for_slice, \
    get_trajs_from_wandb, get_quad_approx_reward_for_slice

matplotlib.rc('font', size=10)

slice_dict = {
    'x': ['u', 'u', 'v', 'v'],
    'y': ['u', 'v', 'u', 'v']
}
slice_list = [{'x': sx, 'y': sy} for sx, sy in zip(slice_dict['x'], slice_dict['y'])]
slice_lists = [[slice_list[0], slice_list[1]],
               [slice_list[2], slice_list[3]]]

n_layers = 5
num_points_per_side = 20


A = onp.array([[0.511, 0.064],
               [0.533, 0.993]])
lq_params = get_default_lq_params(A=A, b=0., r=0.01, q=0.147)
lq_game = GeneralSumLinearQuadraticGame(lq_params)

trajs_all, runs = get_trajs_from_wandb(filter_tags=['plot_quad_approx'])

for run_i in range(len(runs)):
    trajs_dict = trajs_all[run_i]
    run = runs[run_i]

    len_traj = len(trajs_dict['u']['x'])
    total_steps = len_traj * run.config['log_every']

    Ku_center = onp.array([[trajs_dict['u']['x'][0], trajs_dict['u']['y'][0]]])
    Kv_center = onp.array([[trajs_dict['v']['x'][0], trajs_dict['v']['y'][0]]])

    _, ravel_fn_ku = ravel_pytree(Ku_center)
    _, ravel_fn_kv = ravel_pytree(Kv_center)

    centers_dict = {
        'u': {'x': Ku_center[0, 0], 'y': Ku_center[0, 1]},
        'v': {'x': Kv_center[0, 0], 'y': Kv_center[0, 1]}
    }

    paddings_dict = get_paddings_from_trajs(centers_dict, trajs_dict)

    plt.figure()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 12), subplot_kw=dict(projection="3d"))

    for layer_i in range(n_layers):
        if layer_i == n_layers - 1:
            alpha = 0.7
        else:
            alpha = 0.3
        step_i = (len_traj - 1) // (n_layers - 1) * layer_i
        print(f"step_i = {step_i}")

        curr_k_dict = {}
        for player in trajs_dict.keys():
            curr_k_dict[player] = {}
            for axis in trajs_dict[player].keys():
                curr_k_dict[player][axis] = trajs_dict[player][axis][step_i]

        curr_ku_flat = onp.array([curr_k_dict['u']['x'], curr_k_dict['u']['y']])
        curr_kv_flat = onp.array([curr_k_dict['v']['x'], curr_k_dict['v']['y']])
        j00, j01, j10, j11 = lq_game.compute_jacobian(cov_rollout_steps=30,
                                                      ravel_fn_ku=ravel_fn_ku, ravel_fn_kv=ravel_fn_kv,
                                                      Ku_flat=curr_ku_flat, Kv_flat=curr_kv_flat)
        jac = onp.vstack([onp.hstack([j00, j01]),
                          onp.hstack([j10, j11])])

        for plot_i in range(2):
            for plot_j in range(2):
                slice2d = slice_lists[plot_i][plot_j]
                meshes = get_meshes_for_slice(centers_dict, curr_k_dict, slice2d, paddings_dict, num_points_per_side)
                reward_mesh = get_quad_approx_reward_for_slice(jac, meshes, num_points_per_side, centers=curr_k_dict)

                # z axis
                axes[plot_i][plot_j].contourf(meshes[slice2d['x']]['x'],
                                              meshes[slice2d['y']]['y'],
                                              reward_mesh,
                                              offset=(step_i + 1) * run.config['log_every'], alpha=alpha)

    for plot_i in range(2):
        for plot_j in range(2):
            slice2d = slice_lists[plot_i][plot_j]

            traj = get_traj_for_slice(trajs_dict, slice2d)

            # z axis
            axes[plot_i][plot_j].plot(traj['x'], traj['y'], onp.linspace(1, total_steps, len_traj),
                                      color='red', linewidth=3)
            axes[plot_i][plot_j].scatter(traj['x'][-1], traj['y'][-1], total_steps,
                                         marker='*', alpha=1, s=100, color='red')

            axes[plot_i][plot_j].set_xlabel(f"K{slice2d['x']}_1", fontsize=14)
            axes[plot_i][plot_j].set_ylabel(f"K{slice2d['y']}_2", fontsize=14)
            axes[plot_i][plot_j].set_zlabel(f"Steps", fontsize=14)

            axes[plot_i][plot_j].set_zlim([1, total_steps])
            xlim = [
                centers_dict[slice2d['x']]['x'] - paddings_dict[slice2d['x']]['x'],
                centers_dict[slice2d['x']]['x'] + paddings_dict[slice2d['x']]['x'],
            ]
            ylim = [
                centers_dict[slice2d['y']]['y'] - paddings_dict[slice2d['y']]['y'],
                centers_dict[slice2d['y']]['y'] + paddings_dict[slice2d['y']]['y'],
            ]
            axes[plot_i][plot_j].set_xlim(xlim)
            axes[plot_i][plot_j].set_ylim(ylim)

            axes[plot_i][plot_j].locator_params(axis="x", nbins=2)
            axes[plot_i][plot_j].locator_params(axis="y", nbins=2)
            axes[plot_i][plot_j].ticklabel_format(axis='z', style='sci', scilimits=(0, 0))

    if run.config['algo'] == 'GDA':
        azim_angle = 210
    else:
        azim_angle = 30
    axes[0, 0].view_init(elev=30., azim=azim_angle)
    axes[0, 1].view_init(elev=30., azim=30)
    axes[1, 0].view_init(elev=30., azim=30)
    axes[1, 1].view_init(elev=30., azim=30)
    plt.tight_layout()
    plt.savefig(f'plots/deterministic/quad_approx_{run.config["algo"]}_{run.name}.pdf')
    plt.show()

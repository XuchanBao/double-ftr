import matplotlib
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
import numpy as onp

from deterministic.utils import get_default_lq_params
from deterministic.lq_game_gs import GeneralSumLinearQuadraticGame
from deterministic.visualization.plot_utils import get_centers_from_trajs, get_paddings_from_trajs, \
    get_meshes_for_slice, get_traj_for_slice, get_trajs_from_wandb, get_rewards_mesh_for_slice

matplotlib.rc('font', size=10)

slice_dict = {
    'x': ['u', 'v'],
    'y': ['u', 'v']
}
slice_list = [{'x': sx, 'y': sy} for sx, sy in zip(slice_dict['x'], slice_dict['y'])]
slice_lists = [slice_list[0], slice_list[1]]

n_layers = 5
num_points_per_side = 20
padding_multiplier = 1.5

A = onp.array([[0.511, 0.064],
               [0.533, 0.993]])
lq_params = get_default_lq_params(A=A, b=0., r=0.01, q=0.147)
lq_game = GeneralSumLinearQuadraticGame(lq_params)

ku = onp.zeros((1, 2))
kv = onp.zeros((1, 2))
_, ravel_fn_ku = ravel_pytree(ku)
_, ravel_fn_kv = ravel_pytree(kv)

trajs_all, runs = get_trajs_from_wandb(filter_tags=['plot_rewards'])

for run_i in range(len(runs)):
    trajs_dict = trajs_all[run_i]
    run = runs[run_i]

    len_traj = len(trajs_dict['u']['x'])
    total_steps = len_traj * run.config['log_every']

    centers_dict = get_centers_from_trajs(trajs_dict)

    paddings_dict = get_paddings_from_trajs(centers_dict, trajs_dict, padding_multiplier)

    plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), subplot_kw=dict(projection="3d"))

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
            slice2d = slice_lists[plot_i]
            meshes = get_meshes_for_slice(centers_dict, curr_k_dict, slice2d, paddings_dict, num_points_per_side)

            reward_meshes = get_rewards_mesh_for_slice(lq_game, meshes, num_points_per_side)

            # z axis
            axes[plot_i].contourf(meshes[slice2d['x']]['x'],
                                  meshes[slice2d['y']]['y'],
                                  reward_meshes[slice2d['x']],
                                  offset=(step_i + 1) * run.config['log_every'], alpha=alpha)

    for plot_i in range(2):
        slice2d = slice_lists[plot_i]

        traj = get_traj_for_slice(trajs_dict, slice2d)

        # z axis
        axes[plot_i].plot(traj['x'], traj['y'], onp.linspace(1, total_steps, len_traj),
                          color='red', linewidth=3)
        axes[plot_i].scatter(traj['x'][-1], traj['y'][-1], total_steps,
                             marker='*', alpha=1, s=100, color='red')

        axes[plot_i].set_xlabel(f"K{slice2d['x']}_1", fontsize=14)
        axes[plot_i].set_ylabel(f"K{slice2d['y']}_2", fontsize=14)
        axes[plot_i].set_zlabel(f"Steps", fontsize=14)

        axes[plot_i].set_zlim([1, total_steps])
        xlim = [
            centers_dict[slice2d['x']]['x'] - paddings_dict[slice2d['x']]['x'],
            centers_dict[slice2d['x']]['x'] + paddings_dict[slice2d['x']]['x'],
        ]
        ylim = [
            centers_dict[slice2d['y']]['y'] - paddings_dict[slice2d['y']]['y'],
            centers_dict[slice2d['y']]['y'] + paddings_dict[slice2d['y']]['y'],
        ]
        axes[plot_i].set_xlim(xlim)
        axes[plot_i].set_ylim(ylim)

        axes[plot_i].locator_params(axis="x", nbins=2)
        axes[plot_i].locator_params(axis="y", nbins=2)
        axes[plot_i].ticklabel_format(axis='z', style='sci', scilimits=(0, 0))

    axes[0].view_init(elev=30., azim=120)
    axes[1].view_init(elev=30., azim=300)
    plt.tight_layout()
    plt.savefig(f'plots/deterministic/rewards_uu_vv_{run.config["algo"]}_{run.name}.pdf')
    plt.show()

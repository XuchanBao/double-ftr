import autograd.numpy as np


def run_optimizer(func, x0, y0, lrx, lry, n_iter, optim_step):
    curr_x = x0
    curr_y = y0

    trajectory = [[curr_x, curr_y]]

    for itr_i in range(n_iter):
        delta_x, delta_y = optim_step(curr_x, curr_y, lrx, lry)

        curr_x -= delta_x
        curr_y -= delta_y

        trajectory.append([curr_x.item(), curr_y.item()])

    return np.array(trajectory)

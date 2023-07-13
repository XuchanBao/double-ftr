import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import wandb

import tensorflow as tf
import numpy as np

from tqdm import tqdm

from utils.data import MOG_1D, MOG_2D
from utils.misc import *
from utils.optim import RMSProp
from utils.logger import get_logger
from utils.train_utils import generator, discriminator, activation_fn, init_plotting, \
    load_session, sample_batch, get_log_dir_name, opt

print(opt)

default_params = vars(opt)
default_params['damping_min_log10'] = np.log10(opt.damping_min)
tags = []
run = wandb.init(project=f"ftr-{opt.data}", entity='follow-the-ridge', config=default_params, tags=tags)

# fix random seed for np
np.random.seed(opt.seed)

name = get_log_dir_name()

root_dir = 'results/' + name
os.makedirs(root_dir, exist_ok=True)

init_plotting()

# create logger
path = os.path.dirname(os.path.abspath(__file__))
path_file = os.path.join(path, 'follow_ridge.py')
package_file = os.path.join(path, 'utils/misc.py')
logger = get_logger('log', logpath=root_dir + '/', filepath=path_file, package_files=[package_file])
logger.info(vars(opt))

rng = np.random.RandomState(seed=opt.seed)
if opt.data == 'MOG-1D':
    data_generator = MOG_1D(rng, std=opt.data_std)
else:
    data_generator = MOG_2D(opt.data, rng, std=opt.data_std)

tf.reset_default_graph()
tf.set_random_seed(opt.seed)
sess = tf.Session()

real_samples = tf.placeholder(tf.float32, [None, opt.x_dim])
noise = tf.placeholder(tf.float32, [None, opt.z_dim])

# Construct generator and discriminator nets
fake_samples = generator(noise, output_dim=opt.x_dim)
real_score = discriminator(real_samples)
fake_score = discriminator(fake_samples, reuse=True)

# Standard GAN loss
loss_disc_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score,
                                            labels=tf.ones_like(real_score)))
loss_disc_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score,
                                            labels=tf.zeros_like(fake_score)))
loss_disc = loss_disc_real + loss_disc_fake
if opt.loss_fn == 'original':
    loss_gen = -loss_disc_fake  # Saddle objective
elif opt.loss_fn == 'modified':
    loss_gen = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score,
                                                labels=tf.ones_like(fake_score)))
else:
    raise NotImplementedError(f"loss_fn {opt.loss_fn} is not supported!")

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

# regularization
loss_disc += opt.weight_decay * tf.reduce_sum(flatten(disc_vars) ** 2)
loss_gen += opt.weight_decay_g * tf.reduce_sum(flatten(gen_vars) ** 2)

# misc functions
gen_set_params = SetFromFlat(sess, gen_vars)
gen_get_params = GetFlat(sess, gen_vars)

disc_set_params = SetFromFlat(sess, disc_vars)
disc_get_params = GetFlat(sess, disc_vars)

disc_unflatten = unflatten(disc_vars)
gen_unflatten = unflatten(gen_vars)

# create optimizers
g_opt = RMSProp(opt.gen_learning_rate, gen_vars, name='g_rmsprop')
d_opt = RMSProp(opt.disc_learning_rate, disc_vars, name='d_rmsprop')

# damping parameters
damping = tf.Variable(opt.damping, name="damping_d", dtype=tf.float32, trainable=False)
damping_gen = tf.Variable(opt.damping, name="damping_g", dtype=tf.float32, trainable=False)

new_damp_val = tf.placeholder(tf.float32, shape=())
min_damp_val = tf.placeholder(tf.float32, shape=())

assign_damp_d = tf.assign(damping, new_damp_val)
assign_damp_g = tf.assign(damping_gen, new_damp_val)
assign_min_damp_d = tf.assign(damping, tf.minimum(damping, min_damp_val))
assign_min_damp_g = tf.assign(damping_gen, tf.minimum(damping_gen, min_damp_val))

# gradient operator
gen_grads = tf.gradients(loss_gen, gen_vars)
gen_grads_flat = flatten(gen_grads)
disc_grads = tf.gradients(loss_disc, disc_vars)
disc_grads_flat = flatten(disc_grads)

# hessian-vector product
vecs_d = tf.placeholder(tf.float32, [None])
unflatten_vecs_d = disc_unflatten(vecs_d)
vecs_g = tf.placeholder(tf.float32, [None])
unflatten_vecs_g = gen_unflatten(vecs_g)
hvp = flatten(tf.gradients(disc_grads, disc_vars, grad_ys=unflatten_vecs_d))
hvp_gen = flatten(tf.gradients(gen_grads, gen_vars, grad_ys=unflatten_vecs_g))

d_num_params = flatten(disc_vars).get_shape().as_list()[0]
g_num_params = flatten(gen_vars).get_shape().as_list()[0]

# gradient norm
gradient_norm_g = tf.reduce_sum(gen_grads_flat ** 2)
gradient_norm_d = tf.reduce_sum(disc_grads_flat ** 2)

# preconditioning
gen_interp_coeff = disc_interp_coeff = None
if opt.vanilla_gradient:
    gen_pred_grads_flat = gen_grads_flat
    disc_pred_grads_flat = disc_grads_flat
else:
    gen_pred_grads_flat = flatten(g_opt.preconditioning(list(zip(gen_grads, gen_vars))))
    disc_pred_grads_flat = flatten(g_opt.preconditioning(list(zip(disc_grads, disc_vars))))

    if opt.precond_interpolate:
        gen_interp_coeff = interpolation_coeff(gradient_norm_g)
        disc_interp_coeff = interpolation_coeff(gradient_norm_d)
        gen_pred_grads_flat = (1. - gen_interp_coeff) * gen_pred_grads_flat + gen_interp_coeff * gen_grads_flat
        disc_pred_grads_flat = (1. - disc_interp_coeff) * disc_pred_grads_flat + disc_interp_coeff * disc_grads_flat

# define saver
# model variables including rmsprop (non-trainable) variables
gen_vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator")
disc_vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator")
saver = tf.train.Saver(
    var_list=gen_vars_all + disc_vars_all + [g_opt._beta_power, d_opt._beta_power, damping, damping_gen],
    max_to_keep=1000)

# training
sess.run(tf.global_variables_initializer())

load_session(sess, saver)

init_gen_params = gen_get_params()
init_disc_params = disc_get_params()

gnorm_list = []
dnorm_list = []


def single_update(itr, old_gen_velocity, old_disc_velocity, plot=False):
    global damping
    global damping_gen
    feed_dict = sample_batch(rng, data_generator, real_samples, noise)

    def hessian_vector_prod(p):
        feed_dict_ = {**feed_dict, **{vecs_d: p}}
        return sess.run(hvp, feed_dict=feed_dict_)

    def hessian_vector_prof_gen(p):
        feed_dict_ = {**feed_dict, **{vecs_g: p}}
        return sess.run(hvp_gen, feed_dict=feed_dict_)

    g_grads = sess.run(gen_pred_grads_flat, feed_dict=feed_dict)

    # old_gen_params = gen_get_params()
    # gen_set_params(old_gen_params - opt.gen_learning_rate * g_grads)
    d_grads = sess.run(disc_pred_grads_flat, feed_dict=feed_dict)
    # gen_set_params(old_gen_params)

    gen_gd_update = gen_velocity = opt.gen_learning_rate * g_grads + opt.momentum * old_gen_velocity
    disc_gd_update = disc_velocity = opt.disc_learning_rate * d_grads + opt.momentum * old_disc_velocity

    if opt.follow_ridge:
        # compute the correction term in FR
        old_gen_params = gen_get_params()
        old_disc_grads = sess.run(disc_grads_flat, feed_dict)
        gen_set_params(old_gen_params - gen_gd_update)
        new_disc_grads = sess.run(disc_grads_flat, feed_dict)
        e_grads = (old_disc_grads - new_disc_grads) / opt.gen_learning_rate

        # take the Hessian inverse
        damp_d_val = sess.run(damping)
        pred_e_grads = conjugate_gradient(hessian_vector_prod, e_grads, damp_d_val, max_iter=opt.inner_iter)
        pred_e_grads = np.nan_to_num(pred_e_grads)

        if opt.adapt_damping:
            reference = opt.gen_learning_rate ** 2 * np.sum(e_grads ** 2) + 1e-16
            q_model = (hessian_vector_prod(pred_e_grads) - e_grads) * opt.gen_learning_rate
            q_ratio = (reference - np.sum(q_model ** 2)) / reference

            old_disc_grads = sess.run(disc_grads_flat, feed_dict)
            old_disc_params = disc_get_params()
            disc_set_params(old_disc_params + opt.gen_learning_rate * pred_e_grads)
            new_disc_grads = sess.run(disc_grads_flat, feed_dict)
            disc_set_params(old_disc_params)
            true_model = new_disc_grads - old_disc_grads - opt.gen_learning_rate * e_grads
            true_ratio = (reference - np.sum(true_model ** 2)) / reference

            rel_ratio = true_ratio / (q_ratio + 1e-16)
            if itr % 100 == 0:
                logger.info('==> Iteration: %d' % itr)
                logger.info('==> Damping: %e' % damp_d_val)
                logger.info('\n')

            if opt.adapt_damping:
                if rel_ratio < opt.adapt_damping_thres[0] or true_ratio < opt.adapt_damping_thres[0]:
                    pred_e_grads = 0.0 * pred_e_grads
                    # sess.run(assign_damp_d, feed_dict={new_damp_val: damp_d_val * opt.adapt_damping_coeff[0]})
                    sess.run(assign_damp_d, feed_dict={new_damp_val: opt.damping})
                    # damping = damping * 2
                elif rel_ratio < opt.adapt_damping_thres[1]:
                    sess.run(assign_damp_d, feed_dict={new_damp_val: damp_d_val * opt.adapt_damping_coeff[1]})
                    # damping = damping * 1.1
                elif rel_ratio > opt.adapt_damping_thres[2]:
                    # damping = max(1e-8, damping * 0.9)
                    sess.run(assign_damp_d,
                             feed_dict={new_damp_val: max(opt.damping_min, damp_d_val * opt.adapt_damping_coeff[2])})
                # damping = min(damping, opt.damping)
                sess.run(assign_min_damp_d, feed_dict={min_damp_val: opt.damping})

                if itr % opt.log_every == 0:
                    wandb.log({"Damping-D/damping_val": damp_d_val,
                               "Damping-D/true_ratio": true_ratio,
                               "Damping-D/q_ratio": q_ratio,
                               "Damping-D/rel_ratio": rel_ratio,
                               "Damping-D/reference": reference}, step=itr)

        # compute the gradient of disc at an extrapolated point
        old_disc_params = disc_get_params()
        disc_set_params(old_disc_params + opt.gen_learning_rate * pred_e_grads)
        d_grads = sess.run(disc_pred_grads_flat, feed_dict=feed_dict)
        disc_set_params(old_disc_params)
        disc_update = opt.disc_learning_rate * d_grads

        # reset old generator parameters
        gen_set_params(old_gen_params)

        disc_velocity = opt.momentum * old_disc_velocity + disc_update
        disc_update = disc_velocity - opt.gen_learning_rate * pred_e_grads

    else:
        disc_update = disc_gd_update

    if opt.double_ftr:
        # compute the correction term in double FR (generator)
        old_disc_params = disc_get_params()
        old_gen_grads = sess.run(gen_grads_flat, feed_dict)
        disc_set_params(old_disc_params - disc_gd_update)
        new_gen_grads = sess.run(gen_grads_flat, feed_dict)
        e_grads_gen = (old_gen_grads - new_gen_grads) / opt.disc_learning_rate

        # take the Hessian inverse
        damp_g_val = sess.run(damping_gen)
        pred_e_grads_gen = conjugate_gradient(hessian_vector_prof_gen, e_grads_gen, damp_g_val, max_iter=opt.inner_iter)
        pred_e_grads_gen = np.nan_to_num(pred_e_grads_gen)

        if opt.adapt_damping:
            reference = opt.disc_learning_rate ** 2 * np.sum(e_grads_gen ** 2) + 1e-16
            q_model = (hessian_vector_prof_gen(pred_e_grads_gen) - e_grads_gen) * opt.disc_learning_rate
            q_ratio = (reference - np.sum(q_model ** 2)) / reference

            old_gen_grads = sess.run(gen_grads_flat, feed_dict)
            old_gen_params = gen_get_params()
            gen_set_params(old_gen_params + opt.disc_learning_rate * pred_e_grads_gen)
            new_gen_grads = sess.run(gen_grads_flat, feed_dict)
            gen_set_params(old_gen_params)
            true_model = new_gen_grads - old_gen_grads - opt.disc_learning_rate * e_grads_gen
            true_ratio = (reference - np.sum(true_model ** 2)) / reference

            rel_ratio = true_ratio / (q_ratio + 1e-16)
            if itr % 100 == 0:
                logger.info('==> Iteration: %d' % itr)
                logger.info('==> Damping (G): %e' % damp_g_val)
                logger.info('\n')

            if opt.adapt_damping:
                if rel_ratio < opt.adapt_damping_thres[0] or true_ratio < opt.adapt_damping_thres[0]:
                    pred_e_grads_gen = 0.0 * pred_e_grads_gen
                    # damping_gen = damping_gen * 2
                    sess.run(assign_damp_g, feed_dict={new_damp_val: damp_g_val * opt.adapt_damping_coeff[0]})
                    # sess.run(assign_damp_g, feed_dict={new_damp_val: opt.damping})
                elif rel_ratio < opt.adapt_damping_thres[1]:
                    # damping_gen = damping_gen * 1.1
                    sess.run(assign_damp_g, feed_dict={new_damp_val: damp_g_val * opt.adapt_damping_coeff[1]})
                elif rel_ratio > opt.adapt_damping_thres[2]:
                    # damping_gen = max(1e-8, damping_gen * 0.9)
                    sess.run(assign_damp_g,
                             feed_dict={new_damp_val: max(opt.damping_min, damp_g_val * opt.adapt_damping_coeff[2])})
                # damping_gen = min(damping_gen, opt.damping)
                sess.run(assign_min_damp_g, feed_dict={min_damp_val: opt.damping})

                if itr % opt.log_every == 0:
                    wandb.log({"Damping-G/damping_val": damp_g_val,
                               "Damping-G/true_ratio": true_ratio,
                               "Damping-G/q_ratio": q_ratio,
                               "Damping-G/rel_ratio": rel_ratio,
                               "Damping-G/reference": reference}, step=itr)

        # compute the gradient of gen at an extrapolated point
        old_gen_params = gen_get_params()
        gen_set_params(old_gen_params + opt.disc_learning_rate * pred_e_grads_gen)
        g_grads = sess.run(gen_pred_grads_flat, feed_dict=feed_dict)
        gen_set_params(old_gen_params)
        gen_update = opt.gen_learning_rate * g_grads

        # reset old discriminator parameters
        disc_set_params(old_disc_params)

        gen_velocity = opt.momentum * old_gen_velocity + gen_update
        gen_update = gen_velocity - opt.disc_learning_rate * pred_e_grads_gen
    else:
        gen_update = gen_gd_update

    if itr % opt.log_every == 0:
        gnorm2, dnorm2, n_params_g, n_params_d = sess.run(
            [gradient_norm_g, gradient_norm_d, tf.size(gen_grads_flat), tf.size(disc_grads_flat)], feed_dict=feed_dict)

        log_dict = {"Grad-norm/g2": gnorm2,
                    "Grad-norm/d2": dnorm2,
                    "Grad-norm/g-per-param": np.sqrt(gnorm2) / float(n_params_g),
                    "Grad-norm/d-per-param": np.sqrt(dnorm2) / float(n_params_d),
                    "Grad-norm/g-precond-per-param": np.linalg.norm(g_grads) / float(n_params_g),
                    "Grad-norm/d-precond-per-param": np.linalg.norm(d_grads) / float(n_params_d)}
        if gen_interp_coeff is not None:
            gcoeff, dcoeff = sess.run([gen_interp_coeff, disc_interp_coeff], feed_dict=feed_dict)
            log_dict["Grad-norm/g-interp-coeff"] = gcoeff
            log_dict["Grad-norm/d-interp-coeff"] = dcoeff

        wandb.log(log_dict, step=itr)

    if itr % 100 == 0 and plot:
        gnorm, dnorm = sess.run([gradient_norm_g, gradient_norm_d], feed_dict)
        gnorm_list.append(gnorm)
        dnorm_list.append(dnorm)

        plt.close()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(gnorm_list)
        ax1.set_yscale('log')
        ax2.plot(dnorm_list)
        ax2.set_yscale('log')
        plt.tight_layout()
        plt.savefig(root_dir + '/norm.pdf')

    return gen_update, disc_update, gen_velocity, disc_velocity


# training loop
g_velocity = d_velocity = 0.0
for iteration in tqdm(range(opt.iteration + 1)):
    old_g_params = gen_get_params()
    old_d_params = disc_get_params()

    g_update, d_update, g_velocity, d_velocity = single_update(iteration, g_velocity, d_velocity, plot=True)
    gen_set_params(old_g_params - g_update)
    disc_set_params(old_d_params - d_update)

    if iteration % 1000 == 0:
        print("====> iteration: %d" % iteration)
        save_path = saver.save(sess, f"{root_dir}/model-opt-save", global_step=iteration)
        print(f">>>>>> Model saved to {save_path}\n")

        samples = sess.run(fake_samples, feed_dict=sample_batch(rng, data_generator, real_samples, noise))
        plt.close()

        if opt.x_dim == 1:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
            sns.kdeplot(np.reshape(samples, [-1]), shade=True, ax=ax1)
            ax1.set_title('Generator Density')
            ax1.set_xlim([-6.0, 6.0])
            ax1.grid(linestyle='--')

            xx = np.linspace(-6, 6, 400)
            xz = sess.run(real_score, feed_dict={real_samples: np.reshape(xx, [-1, 1])})
            xz = 1. / (1. + np.exp(-xz))
            ax2.plot(xx, np.reshape(xz, [-1]), linewidth=2.5)
            ax2.set_title('Discriminator Prediction')
            ax2.set_ylim([0.0, 1.0])
            ax2.grid(linestyle='--')

            xx = np.linspace(0, iteration, (iteration // 100) + 1)
            line1, = ax3.plot(xx, dnorm_list, linewidth=2.5)
            line2, = ax3.plot(xx, gnorm_list, linewidth=2.5)
            ax3.set_title('Gradient norm')
            ax3.set_yscale('log')
            ax3.legend((line1, line2), ('discriminator', 'generator'))
            ax3.grid(linestyle='--')
            plt.tight_layout()

            wandb.log({"Visualization": wandb.Image(plt)}, step=iteration)
            plt.savefig(root_dir + '/iter-%d.pdf' % iteration)
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
            sns.kdeplot(samples[:, 0], samples[:, 1], shade=True, cmap='Reds', ax=ax1)
            ax1.set_xlim([-5, 5])
            ax1.set_ylim([-5, 5])
            ax1.set_title('Generator Density')

            x = np.linspace(-5, 5, 64)
            y = np.linspace(-5, 5, 64)
            X, Y = np.meshgrid(x, y)
            xy = np.concatenate((np.reshape(X, [-1, 1]), np.reshape(Y, [-1, 1])), axis=1)
            z = sess.run(real_score, feed_dict={real_samples: xy})
            z = 1. / (1. + np.exp(-z))
            Z = np.reshape(z, [64, 64])
            c = ax2.pcolor(X, Y, Z, cmap='Blues')
            f.colorbar(c, ax=ax2)
            ax2.set_title('Discriminator Prediction')
            plt.tight_layout()

            wandb.log({"Visualization": wandb.Image(plt)}, step=iteration)
            plt.savefig(root_dir + '/iter-%d.pdf' % iteration)

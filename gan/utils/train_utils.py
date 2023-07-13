import argparse
import tensorflow as tf
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--iteration", type=int, default=50000, help="number of iterations of training")
parser.add_argument("--loss_fn", type=str, default='original', help='the GAN loss function for the generator')
parser.add_argument("--batch_size", type=int, default=5000, help="size of the batches")
parser.add_argument("--gen_learning_rate", "--g_lr", type=float, default=0.0002, help="generator learning rate")
parser.add_argument("--disc_learning_rate", "--d_lr", type=float, default=0.0002,
                    help="discriminator learning rate")
parser.add_argument("--weight_decay", "--wd", type=float, default=0.0001, help="weight decay coefficient for disc")
parser.add_argument("--weight_decay_g", "--wd_g", type=float, default=0.0, help="weight decay coefficient for gen")
parser.add_argument("--z_dim", type=int, default=16, help="dimension of latent node")
parser.add_argument("--g_hidden", type=int, default=64, help="dimension of hidden units")
parser.add_argument("--d_hidden", type=int, default=64, help="dimension of hidden units")
parser.add_argument("--residual_g", action='store_true', help="whether to use residual connection for generator")
parser.add_argument("--residual_d", action='store_true',
                    help="whether to use residual connection for discriminator")
parser.add_argument("--layernorm", action='store_true', help="whether to use layernorm (both D and G)")
parser.add_argument("--d_layers", type=int, default=2, help="num of hidden layer")
parser.add_argument("--g_layers", type=int, default=2, help="num of hidden layer")
parser.add_argument("--x_dim", type=int, help="data dimension")
parser.add_argument("--momentum", type=float, default=0.0, help="momentum coefficient for the whole system")
parser.add_argument("--init", type=str, default="xavier", help="which initialization scheme")
parser.add_argument("--data", type=str, help="which dataset")
parser.add_argument("--data_std", type=float, default=0.1, help="std for each mode")
parser.add_argument("--g_act", type=str, default="tanh", help="which activation function for gen")
parser.add_argument("--d_act", type=str, default="tanh", help="which activation function for disc")

parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', action='store_true', help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt_itr', default=None, type=int, help='Ckpt iteration to restore (default: none (latest))')
parser.add_argument("--vanilla_gradient", "--vg", action='store_true', help="whether to remove preconditioning")
parser.add_argument("--follow_ridge", action='store_true', help="whether to use  follow-the-ridge")
parser.add_argument("--double_ftr", action='store_true', help="whether to use FTR for generator too")
parser.add_argument("--inner_iter", type=int, default=5, help="conjugate gradient or gradient descent steps")
parser.add_argument("--damping", type=float, default=1.0, help="initial damping term for CG")
parser.add_argument("--adapt_damping", action='store_true', help="whether to adapt damping")
parser.add_argument("--results_dir_prefix", default="", help="prefix for the folder to save results")
parser.add_argument("--adapt_damping_thres", type=float, nargs="+", default=[0.0, 0.5, 0.95])
parser.add_argument("--adapt_damping_coeff", type=float, nargs="+", default=[2.0, 1.1, 0.9])
parser.add_argument("--damping_min", type=float, default=1e-8)
parser.add_argument("--precond_interpolate", action='store_true',
                    help="whether to interpolate btwn RMSProp and SGD")
parser.add_argument("--log_every", type=int, default=5)
parser.add_argument("--seed", type=int, default=1234, help="random seed")

opt = parser.parse_args()


def get_log_dir_name():
    # automatically setup the name
    name = '%s-std%.2f/' % (opt.data, opt.data_std)
    name += 'bs%d-z%d-g%dh%d-d%dh%d-ga%s-da%s-glr%.5f-dlr%.5f-%s-wd%.4f-mom%.2f' \
            % (opt.batch_size, opt.z_dim, opt.g_layers, opt.g_hidden, opt.d_layers, opt.d_hidden,
               opt.g_act, opt.d_act, opt.gen_learning_rate, opt.disc_learning_rate, opt.init, opt.weight_decay,
               opt.momentum)

    if opt.follow_ridge:
        name += '-cg%d-damp%.3f' % (opt.inner_iter, opt.damping)
        if opt.adapt_damping:
            name += '-ad'

    use_old_hessian = False
    if use_old_hessian:
        name += "-oldhess"

    double_ftr = opt.double_ftr
    if double_ftr:
        name = f"double-{name}"
    name += f"-wdgen{opt.weight_decay_g:.4f}" \
            f"-thres{opt.adapt_damping_thres[0]:.2f}-{opt.adapt_damping_thres[1]:.2f}-{opt.adapt_damping_thres[2]:.2f}" \
            f"-coeff{opt.adapt_damping_coeff[0]:.1f}-{opt.adapt_damping_coeff[1]:.1f}-{opt.adapt_damping_coeff[2]:.1f}"
    # f"-min{np.log10(opt.damping_min):.1f}"

    if opt.loss_fn != 'original':
        name += f"{opt.loss_fn}-"
    name = f"{opt.results_dir_prefix}{name}"

    if opt.residual_g:
        name += '-resg'
    if opt.residual_d:
        name += '-resd'

    if opt.layernorm:
        name += '-ln'

    # load_root = f'results/{name}'
    if opt.resume:
        name += f'-resume-{opt.ckpt_itr}'

    return name


def load_session(sess, saver):
    # resume
    if opt.resume:
        # if opt.ckpt_itr is None:
        #     logger.info('==> Getting selection pattern from checkpoint..')
        #     latest_ckpt = tf.train.latest_checkpoint(opt.resume)
        #     logger.info(latest_ckpt)
        #     saver.restore(sess, latest_ckpt)
        # else:
        assert opt.ckpt_itr is not None
        logger.info(f'==> Loading checkpoint at iteration {opt.ckpt_itr}')
        save_path = f"results/{opt.resume}/model-opt-save-{opt.ckpt_itr}"
        saver.restore(sess, save_path)


def activation_fn(name):
    if name == 'elu':
        return tf.nn.elu
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'tanh':
        return tf.nn.tanh


# set global settings
def init_plotting():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.0 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.0 * plt.rcParams['font.size']


def generator(x, output_dim=1, n_hidden=opt.g_hidden, n_layer=opt.g_layers,
              initializer=(tf.glorot_normal_initializer(seed=opt.seed) if opt.init == 'xavier'
              else tf.initializers.orthogonal(gain=1.0))):
    with tf.variable_scope("generator"):
        activation = activation_fn(opt.g_act)
        ln = tf.keras.layers.LayerNormalization()
        if opt.residual_g:
            pre_act = tf.layers.dense(x, n_hidden, activation=None, kernel_initializer=initializer)

            if opt.layernorm:
                pre_act = ln(pre_act)

            x = activation(pre_act)
            pre_res = x

        for _ in range(n_layer):
            pre_act = tf.layers.dense(x, n_hidden, activation=None, kernel_initializer=initializer)

            if opt.layernorm:
                pre_act = ln(pre_act)

            x = activation(pre_act)

        if opt.residual_g:
            x = activation(pre_res + pre_act)

        x = tf.layers.dense(x, output_dim, activation=None, kernel_initializer=initializer)
    return x


def discriminator(x, n_hidden=opt.d_hidden, n_layer=opt.d_layers, reuse=False,
                  initializer=(tf.glorot_normal_initializer(seed=opt.seed) if opt.init == 'xavier'
                  else tf.initializers.orthogonal(gain=1.0))):
    with tf.variable_scope("discriminator", reuse=reuse):
        activation = activation_fn(opt.d_act)
        ln = tf.keras.layers.LayerNormalization()
        if opt.residual_d:
            pre_act = tf.layers.dense(x, n_hidden, activation=None, kernel_initializer=initializer)
            if opt.layernorm:
                pre_act = ln(pre_act)

            x = activation(pre_act)
            pre_res = x

        for i in range(n_layer):
            pre_act = tf.layers.dense(x, n_hidden, activation=None, kernel_initializer=initializer)
            if opt.layernorm:
                pre_act = ln(pre_act)
            x = activation(pre_act)

        if opt.residual_d:
            x = activation(pre_res + pre_act)
        x = tf.layers.dense(x, 1, activation=None, kernel_initializer=initializer)
    return x


def sample_batch(rng, data_generator, real_samples, noise):
    z = rng.normal(size=[opt.batch_size, opt.z_dim])
    x = data_generator.sample(opt.batch_size)
    feed_dict = {real_samples: x, noise: z}
    return feed_dict


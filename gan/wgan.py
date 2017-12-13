# We import the modules needed.
import sys, os, datetime, time
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import utils.utils as utils
from IPython.display import clear_output
from utils.ops import *

class Wgan(object):
    model_name = "wgan"
    def __init__(self, sess, epoch, bs, z_dim, ds_name):
        self.sess = sess
        self.epoch = epoch
        self.z_dim = z_dim
        self.batch_size = bs
        if ds_name == 'mnist' or ds_name == 'fashionMnist':
            if ds_name == 'mnist':
                print("Loading mnist")
                self.train_dataSet, self.test_dataSet = utils.regMnist()
            else:
                print("Loading fashion mnist")
                self.train_dataSet, self.test_dataSet = utils.fasionMnist()
            self.imgW=28
            self.imgH=28
            self.start_channels = 1
            self.lr = 0.0002
            self.beta1 = 0.5
            self.training_iters = 2 ** 24
            self.display_step = 256
            self.d_steps = 3
            self.g_factor = 3

        if ds_name == 'unnAss':
            print("Loading unnAss")
            self.train_dataSet, self.test_dataSet = utils.load_images_from_folder("../data/NormalizedImages/")
            self.imgW=128
            self.imgH=128
            self.start_channels = 3
            self.lr = 0.0002
            self.beta1 = 0.5
            self.training_iters = 2 ** 20
            self.display_step = 16
            self.d_steps = 3
            self.g_factor = 3

                # tf Graph input


    # Create model using wrappers
    # the model is identical
    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            x = tf.reshape(x, [-1, self.imgW, self.imgH, self.start_channels])
            net = tf.layers.conv2d(x, 64, 4, padding="same", activation=utils.lrelu)
            net = utils.maxpool2d(net, 2)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.layers.conv2d(net, 128, 4, padding="same", activation=utils.lrelu)
            net = utils.maxpool2d(net, 2)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.layers.conv2d(net, 128, 4, padding="same", activation=utils.lrelu)
            net = utils.maxpool2d(net, 2)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.reshape(net, [-1, 128 * 16 * 16])
            net = tf.layers.dense(net, 1024, activation=utils.lrelu)
            out_logits = tf.layers.dense(net, 1)
            out = tf.nn.sigmoid(out_logits)
        return out, out_logits, net

    ## Create model using wrappers
    # the model is identical
    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse = reuse):
            net = tf.layers.dense(z, 1024, activation = tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.layers.dense(net, 128 * 8 * 8, activation=tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.reshape(net, [-1, 8, 8, 128])
            net = tf.depth_to_space(net, 2)
            net = tf.layers.conv2d(net, 64, 4, padding="same", activation=utils.lrelu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.depth_to_space(net, 2)
            net = tf.layers.conv2d(net, 256, 4, padding="same", activation=utils.lrelu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.depth_to_space(net, 2)
            net = tf.layers.conv2d(net, 128, 4, padding="same", activation=utils.lrelu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.depth_to_space(net, 2)
            out = tf.layers.conv2d(net, self.start_channels, 4, padding="same", activation=utils.lrelu)
        return out

    ## Create model using wrappers
    # the model is identical
    def generator2(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse = reuse):
            net = tf.layers.dense(z, 1024, activation = tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.layers.dense(net, 128 * 8 * 8, activation=tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.reshape(net, [-1, 8, 8, 128])
            net = tf.layers.conv2d_transpose(net, 64, 4, strides=2, padding="same", activation=tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.layers.conv2d_transpose(net, 64, 4, strides=2, padding="same", activation=tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.layers.conv2d_transpose(net, 64, 4, strides=2, padding="same", activation=tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_training)
            out = tf.layers.conv2d_transpose(net, self.start_channels, 4, strides=2, padding="same", activation=utils.lrelu)
        return out

    def build_model(self):
        """"""
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None,  self.imgW ,self.imgH,self.start_channels])
            self.z = tf.placeholder(tf.float32, shape=[None, 100], name='z')

        G_sample = self.generator(self.z)
        self.G = G_sample
        D_real, D_real_logits, _ = self.discriminator(self.x)
        D_fake, D_fake_logits, _ = self.discriminator(G_sample, reuse = True)

        with tf.name_scope('loss'):
            d_loss_real = - tf.reduce_mean(D_real_logits)
            d_loss_fake = tf.reduce_mean(D_fake_logits)
            self.D_loss = d_loss_real + d_loss_fake

            # get loss for generator
            self.G_loss = - d_loss_fake

            """ Training """
            # divide trainable variables into a group for D and a group for G
            t_vars = tf.trainable_variables()
            theta_D = [var for var in t_vars if 'discriminator' in var.name]
            theta_G = [var for var in t_vars if 'generator' in var.name]

        """Optimizers"""
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_min = tf.train.AdamOptimizer(self.lr, beta1 = self.beta1).minimize(self.D_loss, var_list = theta_D)
            self.G_min = tf.train.AdamOptimizer(self.lr * self.g_factor, beta1 = self.beta1).minimize(self.G_loss, var_list = theta_G)

        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

        # Add scalars to Tensorboard
        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.D_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)
        # tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.init = tf.global_variables_initializer()

    def train(self):
        # Name for tensorboard.
        name = 'tb_' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M_%S'))
        cwd = os.getcwd()
        tb_path = os.path.join(cwd, "Tensorboard")
        tb_path = os.path.join(tb_path, name)
        self.sess.run(self.init)
        disc_output = [self.D_min, self.clip_D, self.D_loss, self.d_sum]
        gen_output = [self.G_min, self.G_loss, self.g_sum]
        start_time = time.time()
        writer = tf.summary.FileWriter(tb_path, self.sess.graph)
        try:
            stepd = 1
            stepg = 1
            step = 1
            # Keep training until max iterations is reached
            while step * self.batch_size < self.training_iters:

                # load first batch
                for _ in range(self.d_steps):
                    X_mb, _ = self.train_dataSet.next_batch(self.batch_size)
                    feed_dict_disc = {self.x: X_mb, self.z: utils.sample_Z(self.batch_size, self.z_dim)}
                    _, _, D_loss_curr, summaryD = self.sess.run(disc_output, feed_dict=feed_dict_disc)
                    if stepd % self.display_step == 0:
                        writer.add_summary(summaryD, stepd)
                    stepd += 1

                X_mb, _ = self.train_dataSet.next_batch(self.batch_size)
                feed_dict_gen = {self.x: X_mb, self.z: utils.sample_Z(self.batch_size, self.z_dim)}
                _, G_loss_curr, summaryG = self.sess.run(gen_output, feed_dict=feed_dict_gen)
                if stepg % self.display_step == 0:
                    writer.add_summary(summaryG, stepg)
                stepg += 1

                # Testing step see if data is converging
                if step % self.display_step == 0:
                    clear_output()
                    prec = ((step * self.batch_size) / self.training_iters) * 100
                    print("Currently: " + str(prec) + "%")
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (step, self.training_iters, self.batch_size, time.time() - start_time, D_loss_curr, G_loss_curr))

                    samples = self.sess.run(self.G, feed_dict={self.z: utils.sample_Z(4, self.z_dim)})
                    sample = samples[1]
                    sample = (sample - sample.min()) / (sample.max() - sample.min())
                    plt.imshow(sample)
                    plt.show()
                    fig2 = utils.plot(X_mb[:4])
                    plt.show()
                    plt.close(fig2)
                step += 1
            print("\nOptimization Finished!, Training GLOSS = {:.3f}".format(G_loss_curr))

        except KeyboardInterrupt:
            pass

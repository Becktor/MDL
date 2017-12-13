

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

class Cgan(object):
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
            self.input_dim = 3
            self.output_dim = 3
            self.lr = 0.0002
            self.beta1 = 0.5
            self.training_iters = 2 ** 20
            self.display_step = 16
            self.d_steps = 3
            self.g_factor = 3
            self.gdim = 64
            self.dim = 64

                # tf Graph input

    def generator(self, z, reuse=False, name="generator"):
        with tf.variable_scope(name):


            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
                # image is 256 x 256 x input_c_dim

            def residule_block(x, dim, ks=3, s=1, name='res'):
                p = int((ks - 1) / 2)
                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
                print(y)
                y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
                print(y)
                return y + x

            fc1 = tf.layers.dense(z, 128*128, activation=tf.nn.relu)
            image = tf.reshape(fc1, [-1, 128, 128, 1])
            # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
            # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
            # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
            print(image)
            c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            print(c0)
            c1 = tf.nn.relu(instance_norm(conv2d(c0, self.gdim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
            c2 = tf.nn.relu(instance_norm(conv2d(c1, self.gdim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
            c3 = tf.nn.relu(instance_norm(conv2d(c2, self.gdim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))

            # define G network with 9 resnet blocks
            r1 = residule_block(c3, self.gdim * 4, name='g_r1')
            r2 = residule_block(r1, self.gdim * 4, name='g_r2')
            r3 = residule_block(r2, self.gdim * 4, name='g_r3')
            r4 = residule_block(r3, self.gdim * 4, name='g_r4')
            r5 = residule_block(r4, self.gdim * 4, name='g_r5')
            r6 = residule_block(r5, self.gdim * 4, name='g_r6')
            r7 = residule_block(r6, self.gdim * 4, name='g_r7')
            r8 = residule_block(r7, self.gdim * 4, name='g_r8')
            r9 = residule_block(r8, self.gdim * 4, name='g_r9')

            d1 = deconv2d(r9, self.gdim * 2, 3, 2, name='g_d1_dc')
            d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
            d2 = deconv2d(d1, self.gdim, 3, 2, name='g_d2_dc')
            d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
            d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            pred = lrelu(conv2d(d2, self.output_dim, 7, 1, padding='VALID', name='g_pred_c'))
            print(pred)
            return pred

    def discriminator(self, x, reuse=False, name="discriminator"):

        with tf.variable_scope(name):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            image = tf.reshape(x, [-1, self.imgH, self.imgW, self.input_dim])
            h0 = lrelu(conv2d(image, self.dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(instance_norm(conv2d(h0, self.dim * 2, name='d_h1_conv'), 'd_bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, self.dim * 4, name='d_h2_conv'), 'd_bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(instance_norm(conv2d(h2, self.dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
            # h3 is (32 x 32 x self.df_dim*8)
            out_logits = tf.layers.dense(h3, 1)
            out = tf.nn.sigmoid(out_logits)
            return out, out_logits, h3
            # h4 is (32 x 32 x 1)

    def build_model(self):
        """"""
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None,  self.imgW ,self.imgH,self.input_dim])
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

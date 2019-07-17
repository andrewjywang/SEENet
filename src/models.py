import tensorflow as tf
import numpy as np

from src.layers import *


class MotionEncoder(object):
    def __init__(self, name, is_train, motion_dim, activation='leaky', norm='instance'):
        self.name = name
        self.motion_dim = motion_dim
        self.act = activation
        self.norm = norm
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            conv1 = conv_block(inputs, 16, 'ME1', 5, 1, self.reuse, norm=self.norm, activation=self.act,
                               is_train=self.is_train)
            conv2 = conv_block(conv1, 32, 'ME2', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                               is_train=self.is_train)
            conv3 = conv_block(conv2, 64, 'ME3', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                               is_train=self.is_train)
            conv4 = conv_block(conv3, 128, 'ME4', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                               is_train=self.is_train)
            # fc layers
            flatten = tf.layers.flatten(conv4)
            motion_prev = tf.layers.dense(flatten, 512, name='motion_latent_prev', activation=tf.nn.leaky_relu,
                                          reuse=self.reuse)
            motion = tf.layers.dense(motion_prev, self.motion_dim, name='motion_latent', activation=tf.nn.tanh,
                                     reuse=self.reuse)

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return motion


class ContentEncoder(object):
    def __init__(self, name, is_train, content_dim, activation='leaky', norm='instance'):
        self.name = name
        self.content_dim = content_dim
        self.act = activation
        self.norm = norm
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs):

        with tf.variable_scope(self.name, reuse=self.reuse):
            conv1 = conv_block(inputs, 16, 'CE1', 5, 1, self.reuse, norm=self.norm, activation=self.act,
                               is_train=self.is_train)
            conv2 = conv_block(conv1, 32, 'CE2', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                               is_train=self.is_train)
            conv3 = conv_block(conv2, 64, 'CE3', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                               is_train=self.is_train)
            conv4 = conv_block(conv3, 128, 'CE4', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                               is_train=self.is_train)

            # fc layers
            flatten = tf.layers.flatten(conv4)
            content_prev = tf.layers.dense(flatten, 1024, name='content_latent_prev', activation=tf.nn.leaky_relu,
                                           reuse=self.reuse)
            content = tf.layers.dense(content_prev, self.content_dim, name='content_latent',
                                      activation=tf.nn.tanh, reuse=self.reuse)

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return content


class MotionDecoder(object):
    def __init__(self, name, is_train, image_channel, activation='leaky', norm='instance'):
        self.name = name
        self.image_channel = image_channel
        self.act = activation
        self.norm = norm
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            _input = tf.layers.dense(inputs, 512, name='latent_code', activation=tf.nn.leaky_relu,
                                     reuse=self.reuse)
            _input = tf.layers.dense(_input, 16 * 16 * 64, name='latent_code_next', activation=tf.nn.leaky_relu,
                                     reuse=self.reuse)
            deconv0 = tf.reshape(_input, [-1, 16, 16, _input.get_shape().as_list()[1] // 256])
            deconv1 = deconv_block(deconv0, 64, 'D1', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            deconv2 = deconv_block(deconv1, 32, 'D2', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            deconv3 = deconv_block(deconv2, 16, 'D3', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            decode = conv_block(deconv3, self.image_channel, 'decode', 5, 1, self.reuse, norm=None,
                                activation='tanh', is_train=self.is_train, pad='REFLECT')

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return decode


class ContentDecoder(object):
    def __init__(self, name, is_train, image_channel, activation='leaky', norm='instance'):
        self.name = name
        self.image_channel = image_channel
        self.act = activation
        self.norm = norm
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            _input = tf.layers.dense(inputs, 1024, name='latent_code', activation=tf.nn.leaky_relu,
                                     reuse=self.reuse)
            _input = tf.layers.dense(_input, 16 * 16 * 128, name='latent_code_next', activation=tf.nn.leaky_relu,
                                     reuse=self.reuse)
            deconv0 = tf.reshape(_input, [-1, 16, 16, _input.get_shape().as_list()[1] // 256])
            deconv1 = deconv_block(deconv0, 64, 'D1', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            deconv2 = deconv_block(deconv1, 32, 'D2', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            deconv3 = deconv_block(deconv2, 16, 'D3', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            decode = conv_block(deconv3, self.image_channel, 'decode', 5, 1, self.reuse, norm=None,
                                activation='tanh', is_train=self.is_train, pad='REFLECT')

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return decode


class Decoder(object):
    def __init__(self, name, is_train, image_channel, activation='leaky', norm='instance'):
        self.name = name
        self.image_channel = image_channel
        self.act = activation
        self.norm = norm
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            _input = tf.layers.dense(inputs, 1024, name='latent_code', activation=tf.nn.leaky_relu,
                                     reuse=self.reuse)
            _input = tf.layers.dense(_input, 16 * 16 * 128, name='latent_code_next', activation=tf.nn.leaky_relu,
                                     reuse=self.reuse)
            deconv0 = tf.reshape(_input, [-1, 16, 16, _input.get_shape().as_list()[1] // 256])
            deconv1 = deconv_block(deconv0, 64, 'D1', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            deconv2 = deconv_block(deconv1, 32, 'D2', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            deconv3 = deconv_block(deconv2, 16, 'D3', 3, 2, self.reuse, norm=self.norm, activation=self.act,
                                   is_train=self.is_train)
            decode = conv_block(deconv3, self.image_channel, 'decode', 5, 1, self.reuse, norm=None,
                                activation='tanh', is_train=self.is_train, pad='REFLECT')

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return decode


def motion_lstm_cell(name, hidden_size, num_layers):
    with tf.variable_scope(name):
        lstm_cell = lstm_block('lstm_cell', [hidden_size] * num_layers)
        return lstm_cell


class MotionLSTM(object):
    def __init__(self, name, num_layers, hidden_size, output_size, video_length, pred_num, batch_size):
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.video_length = video_length
        self.pred_num = pred_num
        self.batch_size = batch_size
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            lstm_cell = lstm_block('lstm_cell', [self.hidden_size] * self.num_layers, reuse=self.reuse)
            lstm_state = lstm_cell.zero_state(self.batch_size, tf.float32)
            outputs = []
            _inputs = tf.split(inputs, self.video_length - self.pred_num - 1, 0)
            for i in range(self.pred_num):
                output, state = tf.nn.static_rnn(lstm_cell, _inputs, initial_state=lstm_state)
                _output = tf.concat(output,1)
                generated = tf.layers.dense(_output, self.output_size, activation=tf.nn.tanh, name='reshape_dense',
                                            reuse=self.reuse)
                self.reuse = True
                lstm_state = state
                outputs.append(generated)
                _inputs.append(generated)
                _inputs.pop(0)

            output = tf.concat(outputs, 0)
            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return output


class Discriminator(object):
    def __init__(self, name, is_train, norm='instance', activation='leaky'):
        self.name = name
        self.norm = norm
        self.act = activation
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            D = conv_block(inputs, 16, 'D64', 3, 2, self.reuse, self.norm, activation=self.act, is_train=self.is_train)
            D = conv_block(D, 32, 'D128', 3, 2, self.reuse, self.norm, self.act, self.is_train)
            D = conv_block(D, 64, 'D256', 3, 2, self.reuse, self.norm, self.act, self.is_train)
            D = conv_block(D, 128, 'D512', 3, 2, self.reuse, self.norm, self.act, self.is_train)
            D = tf.layers.flatten(D)
            decision = tf.layers.dense(D, 1)

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return tf.nn.sigmoid(decision), decision


class MotionDiscriminator(object):
    def __init__(self, name, num_layers, hidden_size, video_length, batch_size):
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.video_length = video_length
        self.batch_size = batch_size
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            lstm_fw, initial_states_fw = lstm_list('discriminator_fw', [self.hidden_size] * self.num_layers,
                                                   self.batch_size)
            lstm_bw, initial_states_bw = lstm_list('discriminator_bw', [self.hidden_size] * self.num_layers,
                                                   self.batch_size)

            _inputs = tf.split(inputs, self.video_length - 1, 0)
            outputs, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_rnn(lstm_fw, lstm_bw, _inputs,
                                                                                 initial_states_fw=initial_states_fw,
                                                                                 initial_states_bw=initial_states_bw,
                                                                                 dtype=tf.float32)

            output = tf.reshape(tf.concat(outputs, 0), [1, -1])
            decision = tf.layers.dense(output, 1, name='decision_layer', reuse=self.reuse)

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return tf.nn.sigmoid(decision), decision


class Fusion(object):
    def __init__(self, name, latent_size):
        self.name = name
        self.latent_size = latent_size
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            _input = tf.layers.dense(tf.concat([inputs[0], inputs[1]], -1), self.latent_size, name='fusion_layer',
                                     activation=tf.nn.tanh, reuse=self.reuse)

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _input

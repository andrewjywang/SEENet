import tensorflow as tf
import numpy as np


def norm_layer(input, is_train, reuse=True, norm=None):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            eps = 1e-5
            out = tf.contrib.layers.instance_norm(input, epsilon=eps)
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.layers.batch_normalization(input, training=is_train)
    else:
        out = input

    return out


def act_layer(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.nn.leaky_relu(input, alpha=0.2)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    else:
        return input


def conv2d(input, filters_num, filter_size, stride, pad='SAME', dtype=tf.float32, bias=False):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], filters_num]

    w = tf.get_variable('w', filter_shape, dtype, tf.initializers.random_normal(0.0, 1e-4))
    if pad == 'REFLECT':
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    else:
        assert pad in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)

    if bias:
        b = tf.get_variable('b', [1, 1, 1, filters_num], initializer=tf.constant_initializer(0.0))
        conv = conv + b

    return conv


def conv2d_transpose(input, filters_num, filter_size, stride, pad='SAME', dtype=tf.float32):
    assert pad == 'SAME'
    n, h, w, c = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, filters_num, c]
    output_shape = [n, h * stride, w * stride, filters_num]

    w = tf.get_variable('w', filter_shape, dtype, tf.initializers.random_normal(0.0, 1e-4))
    deconv = tf.nn.conv2d_transpose(input, w, output_shape, stride_shape, pad)

    return deconv


def max_pooling(inputs, filter_size, stride=None, padding='VALID'):
    padding = padding.upper()
    shape = [1, filter_size, filter_size, 1]
    if stride is None:
        stride = shape
    else:
        stride = [1, stride, stride, 1]

    return tf.nn.max_pool(inputs, ksize=shape, strides=stride, padding=padding)


def un_pooling(x):
    out = tf.concat(axis=3, values=[x, tf.zeros_like(x)])
    out = tf.concat(axis=2, values=[out, tf.zeros_like(out)])

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        sh = tf.shape(x)
        return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])


def conv_block(inputs, filters_num, name, filter_size, stride, reuse, norm, activation, is_train, pad='SAME',
               bias=False
               ):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d(inputs, filters_num, filter_size, stride, pad, bias=bias)
        out = norm_layer(out, is_train, reuse, norm)
        out = act_layer(out, activation)
        return out


def deconv_block(inputs, filters_num, name, filter_size, stride, reuse, norm, activation, is_train):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d_transpose(inputs, filters_num, filter_size, stride)
        out = norm_layer(out, is_train, reuse, norm)
        out = act_layer(out, activation)
        return out


def res_block(input, filters_num, name, reuse, norm, is_train, pad='REFLECT', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = conv2d(input, filters_num, 3, 1, pad)
            out = norm_layer(out, is_train, reuse, norm)
            out = act_layer(out, 'relu')

        with tf.variable_scope('res2', reuse=reuse):
            out = conv2d(out, filters_num, 3, 1, pad)
            out = norm_layer(out, is_train, reuse, norm)

        with tf.variable_scope('shortcut', reuse=reuse):
            shortcut = conv2d(input, filters_num, 1, 1, reuse, pad, bias=bias)

        return act_layer(shortcut + out, 'relu')


def lstm_block(name, num_layers, dropout_prob=1.0, state_is_tuple=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        cells = []
        for units in num_layers:
            cell = tf.nn.rnn_cell.LSTMCell(units, state_is_tuple=state_is_tuple, reuse=reuse, name='basic_lstm_cell')
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_prob)
            cells.append(cell)

        return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)


def lstm_list(name, num_layers, batch_size, dropout_prob=1.0, state_is_tuple=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        cells = []
        states = []
        for units in num_layers:
            cell = tf.nn.rnn_cell.LSTMCell(units, state_is_tuple=state_is_tuple, reuse=reuse, name='basic_lstm_cell')
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_prob)
            cells.append(cell)
            states.append(cell.zero_state(batch_size, tf.float32))

        return cells, states


def conv_lstm_block(name, num_layers, input_shape, output_channels, dropout_prob=1.0, reuse=False, type='multi'):
    with tf.variable_scope(name, reuse=reuse):
        cells = []
        for idx, kernels in enumerate(num_layers):
            cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=input_shape, output_channels=output_channels,
                                               kernel_shape=kernels)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_prob)
            cells.append(cell)
        if type == 'multi':
            return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        elif type == 'list':
            return cells

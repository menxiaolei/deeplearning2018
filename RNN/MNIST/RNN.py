#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
print('packages imported')

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimgs, trainlabels, testimgs, testlabels = \
    mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
n_train, n_test, dim, nclasses = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]
print(n_train, n_test, dim, nclasses)
print('MNIST loaded')

diminput = 28
dimhidden = 128
dimoutput = nclasses
nsteps = 28

weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
    'output': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'output': tf.Variable(tf.random_normal([dimoutput]))
}

def _RNN(_X, _W, _b, _nsteps, _name):
    # 1.Permute input from [batchsize, nsteps, diminput]
    # to [nsteps, batchsize, diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # 2.Reshape input to [nsteps*batchsize, diminput]
    _X = tf.reshape(_X, [-1, diminput])
    # 3.Input layer => Hidden layer
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 4.Splite data to 'nsteps' chunks.
    _Hsplit = tf.split(_H, _nsteps, 0)
    # 5.get lstm's finnal output
    with tf.variable_scope(_name):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        # _LSTM_O和_LSTM_S保存了每个时间步的输出和状态
        _LSTM_O, _LSTM_S = tf.nn.static_rnn(lstm_cell, _Hsplit, dtype=tf.float32)
    _O = tf.matmul(_LSTM_O[-1], _W['output']) + _b['output']

    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit, 'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O
    }

learnning_rate = 0.001
x = tf.placeholder(tf.float32, [None, nsteps, diminput])
y = tf.placeholder(tf.float32, [None, dimoutput])
myrnn = _RNN(x, weights, biases, nsteps, 'basic')
pred = myrnn['O']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optm = tf.train.GradientDescentOptimizer(learnning_rate).minimize(cost)
accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
init = tf.global_variables_initializer()
print('network ready')

training_epochs = 5
batch_size = 16
display_step = 1
sess = tf.Session()
sess.run(init)
print('Start optmization')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, nsteps, diminput])

        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)/total_batch

    if epoch % display_step == 0:
        testimgs = testimgs.reshape([n_test, nsteps, diminput])
        feeds = {x: testimgs, y: testlabels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print('test acc: %.3f' % test_acc)



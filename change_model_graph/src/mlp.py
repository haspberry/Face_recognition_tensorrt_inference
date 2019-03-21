from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

def inference(images, image_features,hidden1_units, hidden2_units, num_class, regularizer):
    # hidden 1
    with tf.variable_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([image_features, hidden1_units], stddev=1.0 / math.sqrt(float(image_features))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases, name='hidden_output')
        tf.add_to_collection('regularizer_item', regularizer(weights))

    # hidden 2
    with tf.variable_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases, name='hidden_output')
        tf.add_to_collection('regularizer_item', regularizer(weights))

    # linear
    with tf.variable_scope('linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, num_class], stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([num_class]), name='biases')
        logits = tf.add(tf.matmul(hidden2, weights), biases, name = 'logits')
        tf.add_to_collection('regularizer_item', regularizer(weights))

    return logits

def loss(logits, labels):
    with tf.variable_scope('softmax_cross_entropy_loss'):
        labels = tf.to_int64(labels, name = 'labels')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='softmax_cross_entropy_batch')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='softmax_cross_entropy')
        return cross_entropy_mean

def training(loss, learning_rate, global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

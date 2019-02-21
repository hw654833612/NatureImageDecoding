#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:55:30 2018

@author: Wei Huang, Chong Wang
"""

import os
import errno
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import variance_scaling_initializer, batch_norm



"""============================================================================
   CNN-image """
def concatenate(input):
    return np.concatenate(input)
    
def mkdir(path):
    try:
        os.makedirs(path)
        return path
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def confusion_matrix(y_true, y_pred, isaccuracy=False):
    if isaccuracy:
        a = metrics.confusion_matrix(y_true, y_pred)
        n = np.shape(a)
        b = np.repeat(np.reshape(np.sum(a,axis=1), (n[0],1)), n[1], axis=1)
        output = 1.0*a/b
    else:
        output = metrics.confusion_matrix(y_true, y_pred)
    return output
    
def confusion_matrix_heatmap_pd(matrix_data,xlabels,ylabels, title_name, path, name):
    figure=plt.figure(facecolor='w')
    ax=figure.add_subplot(1,1,1,position=[-1.1,-1.1,0.8,0.8])
    ax.set_yticklabels(ylabels); ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(xlabels); ax.set_xticks(range(len(xlabels)))
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    matrix_data_pd = pd.DataFrame(matrix_data, index=xlabels, columns=ylabels)
    sns.heatmap(matrix_data_pd, annot=True, annot_kws={'size':12,'weight':'bold'}, ax=ax, cmap="jet",fmt='.2f', linewidths=1,vmax=1.0, vmin=0.0)
    plt.title(title_name, fontsize='large', fontweight='bold', color='black',loc ='center', verticalalignment='bottom',
              rotation=0, fontstyle='italic', bbox=dict(facecolor='gray', edgecolor='gray', alpha=0.01 ))
    plt.savefig(path + name, dpi=1000)
    plt.show()
    plt.close()
    
def lrelu(input , alpha= 0.2 , name="LeakyReLU"):
    return tf.maximum(input , alpha*input, name)
    
def conv2d(input, output_dim, ksize = [3, 3, 2, 2], padding='SAME', name="conv2d", with_w=False):
    k_h, k_w, d_h, d_w =ksize[0], ksize[1], ksize[2], ksize[3]
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, input.get_shape()[-1], output_dim], initializer= variance_scaling_initializer())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        if padding == 'Other':
            padding = 'VALID'
            input = tf.pad(input, [[0,0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        elif padding == 'VALID':
            padding = 'VALID'
        conv = tf.nn.conv2d(input, w, strides=[1, d_h, d_w, 1], padding=padding)
        shape = conv.get_shape().as_list()
        conv = tf.reshape(tf.nn.bias_add(conv, b), (-1,shape[1],shape[2],shape[3]))
        if with_w:
            return conv, w, b
        else:
            return conv
            
def pool2d(input, k=2, with_max_avg="max"):
    if with_max_avg=="max":
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], data_format='NHWC',padding='SAME', name="maxpool2d")
    elif with_max_avg=="avg":
        return tf.nn.avg_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], data_format='NHWC',padding='SAME', name="avgpool2d")
        
def fully_connect(input, output_size, stddev=0.02, scope=None, with_w=False):
  with tf.variable_scope(scope or "Linear"):
    w = tf.get_variable("weights", [input.get_shape().as_list()[1], output_size], tf.float32, variance_scaling_initializer())
    b = tf.get_variable("biases", [output_size], initializer=tf.constant_initializer(0.0))
    output = tf.matmul(input, w) + b
    if with_w:
        return output, w, b
    else:
        return output
        
def batch_normal(input, scope="scope", reuse=False):
    return batch_norm(input, scale=True, is_training=True, decay=0.9, epsilon=1e-5, updates_collections=None)
    
    
    
"""============================================================================
   DeepLearning-fMRI """
def LSTM_classifier(x,y,n_inputs,n_hidden_units,n_steps,n_classes,number_of_layers,is_training=False):  # X(128 batch, 28 step, 28 inputs)
    x = tf.reshape(x, [-1, n_inputs])
    with tf.variable_scope('lstm_nn_in'):
         W_in = tf.Variable(tf.random_normal([n_inputs,n_hidden_units],stddev=0.01),name="W_in")
         b_in = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]),name="b_in")
    X_in = tf.matmul(x, W_in) + b_in
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    with tf.variable_scope('lstm_nn_out'):
         W_out = tf.Variable(tf.random_normal([n_hidden_units,n_classes],stddev=0.01),name="W_out")
         b_out = tf.Variable(tf.constant(0.1, shape=[n_classes, ]),name="b_out")
    with tf.variable_scope('lstm'):
         lstm_cell=rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
         if is_training==True: lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*number_of_layers)
    init_state = cell.zero_state(tf.shape(X_in)[0], dtype=tf.float32)  # lstm cell is divided into two parts (c_state, h_state)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul(outputs[-1], W_out) + b_out
    if is_training==True:    results=tf.nn.dropout(results,keep_prob=0.5)  
    pred = tf.nn.softmax(results)
    cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(pred),axis=1))
    return pred, cost

def RNN_classifier(x,y,n_inputs,n_hidden_units,n_steps,n_classes,number_of_layers,is_training=False):  # X(128 batch, 28 step, 28 inputs)
    x = tf.reshape(x, [-1, n_inputs])
    with tf.variable_scope('rnn_nn_in'):
         W_in = tf.Variable(tf.random_normal([n_inputs,n_hidden_units],stddev=0.01),name="W_in")
         b_in = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]),name="b_in")
    X_in = tf.matmul(x, W_in) + b_in
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    with tf.variable_scope('rnn_nn_out'):
         W_out = tf.Variable(tf.random_normal([n_hidden_units,n_classes],stddev=0.01),name="W_out")
         b_out = tf.Variable(tf.constant(0.1, shape=[n_classes, ]),name="b_out")
    with tf.variable_scope('rnn'):
         rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden_units)
         if is_training==True: rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=0.5)
    cell = tf.contrib.rnn.MultiRNNCell([rnn_cell]*number_of_layers)
    init_state = cell.zero_state(tf.shape(X_in)[0], dtype=tf.float32)  # lstm cell is divided into two parts (c_state, h_state)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul(outputs[-1], W_out) + b_out
    if is_training==True:    results=tf.nn.dropout(results,keep_prob=0.5)  
    pred = tf.nn.softmax(results)
    cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(pred),axis=1))
    return pred, cost

def GRU_classifier(x,y,n_inputs,n_hidden_units,n_steps,n_classes,number_of_layers,is_training=False):  # X(128 batch, 28 step, 28 inputs)
    x = tf.reshape(x, [-1, n_inputs])
    with tf.variable_scope('gru_nn_in'):
         W_in = tf.Variable(tf.random_normal([n_inputs,n_hidden_units],stddev=0.01),name="W_in")
         b_in = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]),name="b_in")
    X_in = tf.matmul(x, W_in) + b_in
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    with tf.variable_scope('gru_nn_out'):
         W_out = tf.Variable(tf.random_normal([n_hidden_units,n_classes],stddev=0.01),name="W_out")
         b_out = tf.Variable(tf.constant(0.1, shape=[n_classes, ]),name="b_out")
    with tf.variable_scope('gru'):
         gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden_units)
         if is_training==True: gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=0.5)
    cell = tf.contrib.rnn.MultiRNNCell([gru_cell]*number_of_layers)
    init_state = cell.zero_state(tf.shape(X_in)[0], dtype=tf.float32)  # lstm cell is divided into two parts (c_state, h_state)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul(outputs[-1], W_out) + b_out
    if is_training==True:    results=tf.nn.dropout(results,keep_prob=0.5)  
    pred = tf.nn.softmax(results)
    cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(pred),axis=1))
    return pred, cost

def CNN_classifier(x,y,n_steps, n_inputs, n_classes, is_training=False):
    X_in = tf.reshape(x, [-1, n_steps, n_inputs, 1]) # X_in ==> (100 batch, 5 steps, 2000 hidden,1)
    with tf.variable_scope('cnn'):
         conv1 = tf.layers.conv2d(inputs=X_in, filters=256, kernel_size=(5, 1), padding='same', activation=tf.nn.relu)
         conv2 = tf.layers.conv2d(inputs=conv1, filters=256, kernel_size=(5, 1), padding='same', activation=tf.nn.relu)
    conv = tf.reshape(tf.transpose(conv2, [0,1,3,2]), [-1,256*n_steps, n_inputs,1])
    outputs = tf.nn.avg_pool(conv, ksize=[1,256*n_steps,1,1], strides=[1,1,1,1], data_format='NHWC',padding='VALID', name="avgpool2d")
    outputs = tf.squeeze(outputs)
    with tf.variable_scope('cnn_nn_out'):
         W_out = tf.Variable(tf.random_normal([n_inputs,n_classes],stddev=0.01),name="W_out")
         b_out = tf.Variable(tf.constant(0.1, shape=[n_classes, ]),name="b_out")    
    results= tf.matmul(outputs, W_out) + b_out
    if is_training==True:    results=tf.nn.dropout(results,keep_prob=0.5)  
    pred = tf.nn.softmax(results)
    cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(pred),axis=1))
    return pred, cost    
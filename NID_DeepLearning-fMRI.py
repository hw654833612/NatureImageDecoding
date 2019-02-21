#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:55:30 2018

@author: Wei Huang, Chong Wang
"""

import numpy as np
import tensorflow as tf
from NID_utils import RNN_classifier, LSTM_classifier, GRU_classifier, CNN_classifier



sub_name='s_lianlian'
Classifier = 'LSTM'  # options:'LSTM' 'RNN' 'GRU' 'CNN' 


"""============================================================================
   The raw data read constains data and lables. """
Trndata=np.load('G:/LSTM_decoding/'+sub_name+'/Train_23456_2000.npy')
Testdata=np.load('G:/LSTM_decoding/'+sub_name+'/Test_23456_2000.npy')
Valdata=np.load('G:/LSTM_decoding/'+sub_name+'/Validation_23456_2000.npy')
Trnlabels=np.loadtxt('G:/LSTM_decoding/Train_Label.txt')
Testlabels=np.loadtxt('G:/LSTM_decoding/Test_Label.txt')
Vallabels=np.loadtxt('G:/LSTM_decoding/Validation_Label.txt')

Trndata=np.float32(Trndata)
Testdata=np.float32(Testdata)
Valdata=np.float32(Valdata)
Trnlabels=np.float32(Trnlabels)
Testlabels=np.float32(Testlabels)
Vallabels=np.float32(Vallabels)


""" ============================================================================
set parameters """
global_step=tf.Variable(0)
lr = tf.train.exponential_decay(0.0001,global_step,200,0.8,staircase=True) #学习率指数下降
batch_size = 100
n_inputs = 2000  # voxels num
n_steps = 5  # time steps
n_hidden_units = 5000 # neurons in lstm layer
n_classes = 5     # visual stimuli time points 
number_of_layers = 2
samples_num=np.int32(Trndata.shape[0]/n_steps)


""" ============================================================================
placeholder"""
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
is_training = tf.placeholder(tf.bool)

def next_batch(batch_size, TRAIN, label,samples_num):
    idx = np.arange(0, samples_num-1)  # get all possible indexes
    np.random.shuffle(idx)  # shuffle indexes
    TRAIN=np.reshape(TRAIN,[samples_num,n_steps,n_inputs])
    return TRAIN[idx[0:batch_size],:], label[idx[0:batch_size],:]


""" ============================================================================
Classifier"""
if Classifier == 'LSTM':
   pred, cost= LSTM_classifier(x,y,n_inputs=n_inputs,n_hidden_units=n_hidden_units,n_steps=n_steps,n_classes=n_classes,
                                           number_of_layers=number_of_layers,is_training=is_training)
elif Classifier == 'RNN':
   pred, cost= RNN_classifier(x,y,n_inputs=n_inputs,n_hidden_units=n_hidden_units,n_steps=n_steps,n_classes=n_classes,
                                           number_of_layers=number_of_layers,is_training=is_training)
elif Classifier == 'GRU':
   pred, cost= GRU_classifier(x,y,n_inputs=n_inputs,n_hidden_units=n_hidden_units,n_steps=n_steps,n_classes=n_classes,
                                           number_of_layers=number_of_layers,is_training=is_training)
elif Classifier == 'CNN':
   pred, cost= CNN_classifier(x,y,n_inputs=n_inputs,n_steps=n_steps,n_classes=n_classes,is_training=is_training)
train_op = tf.train.AdamOptimizer(lr).minimize(cost,global_step=global_step)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))


""" ============================================================================
run"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())  
min_loss=10
count=0
index=0
test_acc=[]
for epoch in range(1000):
    batch_xs, batch_ys = next_batch(batch_size, Trndata, Trnlabels, samples_num)
    sess.run([train_op], feed_dict={ x:np.reshape(batch_xs,[-1, n_steps, n_inputs]) ,  y:batch_ys,  is_training:True}) 
    acc_test=sess.run(accuracy, feed_dict={x:np.reshape(Testdata,[-1, n_steps, n_inputs]),  y:Testlabels, is_training:False})
    validation_cost=sess.run(cost, feed_dict={x:np.reshape(Valdata,[-1, n_steps, n_inputs]),  y:Vallabels, is_training:False})
    test_acc.append(acc_test)  
    if (epoch+1)%10 == 0:
        print( 'Step: %05d  test accuracy：%.10f \n'%(epoch+1, acc_test))      
    if validation_cost<min_loss:
        min_loss=validation_cost
        index=epoch
        count=0
    else:
        count=count+1
    if count>50:
        break

print(Classifier+'test accuracy： %.10f'%(test_acc[index]))

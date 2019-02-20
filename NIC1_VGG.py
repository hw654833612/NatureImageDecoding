#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:55:30 2018

@author: lembert
"""

import tensorflow as tf
import numpy as np
import random
import time
from hwdlbox.py000_Other import concatenate, roll, resize_ratio
from hwdlbox.py001_Mkdir import mkdir
from hwdlbox.py002_ReadSave import imread, center_crop, transform, get_image, read_image, inverse_transform, merge, imsave, save_images, save_array, read_image_list, save_txt, save_txt_list
from hwdlbox.py003_ClassificationEvaluationIndex import accuracy_score, precision_score, recall_score, f1_score, classification_report, cohen_kappa_score, roc_auc_score, confusion_matrix, confusion_matrix_heatmap, confusion_matrix_heatmap_pd
from hwdlbox.tf000_Others import reshape, conv_cond_concat, MinibatchstateConcat, entropy
from hwdlbox.tf001_ActivationFunction import sigmoid, tanh, relu, lrelu, crelu, relu6, elu, softplus, softsign
from hwdlbox.tf002_LossFunction import l1loss, l2loss, CrossEntropyloss, mseloss
from hwdlbox.tf003_Distance import Euclidean_Distance, Standardized_Euclidean_Distance, Manhattan_Distance, Chebyshev_Distance, Minkowski_Distance, Cosine_Distance, Pearson_Correlation, Hamming_Distance
from hwdlbox.tf004_ConvPoolFCdropoutResNet import conv2d, deconv2d, pool2d, fully_connect, dropout, img_scale, residule_block
from hwdlbox.tf005_Nomalization import batch_normal, instance_norm, Pixl_Norm
from hwdlbox.tf006_LaplacianPyramidGradient import lap_split, lap_split_n, lap_merge, normalize_std, lap_normalize 


""" ======================================================================== """
""" model """
""" ======================================================================== """
keep_prob = 0.1
learning_rate = 0.00001

img    = tf.placeholder(tf.float32, [None, 256, 256, 3])
labels = tf.placeholder(tf.float32, [None, 5])

v1  = batch_normal(lrelu(conv2d(img, 64, [7,7,1,1], padding='SAME', name='conv2d_v1', with_w=False)))
v2  = batch_normal(lrelu(conv2d(v1,  64, [3,3,1,1], padding='SAME', name='conv2d_v2', with_w=False)))
v3  = pool2d(v2, k=2, with_max_avg='max')

v4  = batch_normal(lrelu(conv2d(v3, 128, [3,3,1,1], padding='SAME', name='conv2d_v4', with_w=False)))
v5  = batch_normal(lrelu(conv2d(v4, 128, [3,3,1,1], padding='SAME', name='conv2d_v5', with_w=False)))
v6  = pool2d(v5, k=2, with_max_avg='max')

v7  = batch_normal(lrelu(conv2d(v6, 256, [3,3,1,1], padding='SAME', name='conv2d_v7', with_w=False)))
v8  = batch_normal(lrelu(conv2d(v7, 256, [3,3,1,1], padding='SAME', name='conv2d_v8', with_w=False)))
v9  = batch_normal(lrelu(conv2d(v8, 256, [3,3,1,1], padding='SAME', name='conv2d_v9', with_w=False)))
v10 = pool2d(v9, k=2, with_max_avg='max')

v11  = batch_normal(lrelu(conv2d(v10, 512, [3,3,1,1], padding='SAME', name='conv2d_v11', with_w=False)))
v12  = batch_normal(lrelu(conv2d(v11, 512, [3,3,1,1], padding='SAME', name='conv2d_v12', with_w=False)))
v13  = batch_normal(lrelu(conv2d(v12, 512, [3,3,1,1], padding='SAME', name='conv2d_v13', with_w=False)))
v14 = pool2d(v13, k=2, with_max_avg='max')

v15  = batch_normal(lrelu(conv2d(v14, 512, [3,3,1,1], padding='SAME', name='conv2d_v15', with_w=False)))
v16  = batch_normal(lrelu(conv2d(v15, 512, [3,3,1,1], padding='SAME', name='conv2d_v16', with_w=False)))
v17  = batch_normal(lrelu(conv2d(v16, 512, [3,3,1,1], padding='SAME', name='conv2d_v17', with_w=False)))
v18 = pool2d(v17, k=2, with_max_avg='max')

v19  = batch_normal(lrelu(conv2d(v18, 512, [3,3,1,1], padding='SAME', name='conv2d_v19', with_w=False)))
v20  = batch_normal(lrelu(conv2d(v19, 512, [3,3,1,1], padding='SAME', name='conv2d_v20', with_w=False)))
v21  = batch_normal(lrelu(conv2d(v20, 512, [3,3,1,1], padding='SAME', name='conv2d_v21', with_w=False)))
v22 = pool2d(v21, k=2, with_max_avg='max')

v22_shape = v22.get_shape()
flattened_shape = v22_shape[1].value*v22_shape[2].value*v22_shape[3].value
v23 = tf.reshape(v22, [-1, flattened_shape], name='reshape_v23')
                          
v24 = lrelu(fully_connect(v23, 4096, scope='fc_v24'))
v25 = tf.nn.dropout(v24, keep_prob, name="drop_v25")  

v26 = lrelu(fully_connect(v25, 4096, scope='fc_v26'))
v27 = tf.nn.dropout(v26, keep_prob, name="drop_v27")  

logits = fully_connect(v27, 5, scope='fc_logits')


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)  

y_pred = tf.argmax(logits, 1)
y_true = tf.argmax(labels,1)
correct_pred = tf.equal(y_pred, y_true)  
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  


""" ======================================================================== """
""" load data """
""" ======================================================================== """
Name = 'huangwei'
D_image = np.float32(np.reshape(np.load('../../ExperimentData_Image_fMRI/s_'+Name+'/Stimulus_image.npy'), (2750,256,256,3)))
D_image_label = np.int32(np.reshape(np.load('../../ExperimentData_Image_fMRI/s_'+Name+'/Stimulus_image_label.npy'), (2750,)))


Test_image = np.float32(D_image[0:250,:,:,:])
Test_image_label = D_image_label[0:250]
Test_image_label = (np.arange(6)==Test_image_label[:,None]).astype(np.integer)
Test_image_label = np.delete(Test_image_label, 0, axis=1)


Train_image = np.float32(D_image[250:,:,:,:])
Train_image_label = D_image_label[250:,]
Train_image_label = (np.arange(6)==Train_image_label[:,None]).astype(np.integer)
Train_image_label = np.delete(Train_image_label, 0, axis=1)
del D_image;del D_image_label;



""" ======================================================================== """
""" train """
""" ======================================================================== """
sess = tf.Session()
sess.run(tf.global_variables_initializer())
CM = np.zeros((1000,5,5))
for i in range(10000):
    samples_n = random.sample(np.arange(2500), 36)    
    _, trloss, tracc = sess.run([optimizer, cost, accuracy], feed_dict={img:Train_image[samples_n,:,:,:], labels:Train_image_label[samples_n,:]})  
    if i%10==0:
        teloss1, tecor1, y_pred1, y_true1 = sess.run([cost, correct_pred, y_pred, y_true], feed_dict={img:Test_image[0:50,:,:,:],    labels:Test_image_label[0:50,:]})
        teloss2, tecor2, y_pred2, y_true2 = sess.run([cost, correct_pred, y_pred, y_true], feed_dict={img:Test_image[51:100,:,:,:],  labels:Test_image_label[51:100,:]}) 
        teloss3, tecor3, y_pred3, y_true3 = sess.run([cost, correct_pred, y_pred, y_true], feed_dict={img:Test_image[101:150,:,:,:], labels:Test_image_label[101:150,:]}) 
        teloss4, tecor4, y_pred4, y_true4 = sess.run([cost, correct_pred, y_pred, y_true], feed_dict={img:Test_image[151:200,:,:,:], labels:Test_image_label[151:200,:]}) 
        teloss5, tecor5, y_pred5, y_true5 = sess.run([cost, correct_pred, y_pred, y_true], feed_dict={img:Test_image[201:250,:,:,:], labels:Test_image_label[201:250,:]}) 
        teloss = (teloss1+teloss2+teloss3+teloss4+teloss5)/5.0
        teacc  = np.mean(concatenate((tecor1,tecor2,tecor3,tecor4,tecor5))) 
        Y_pred  = concatenate((y_pred1,y_pred2,y_pred3,y_pred4,y_pred5))
        Y_true  = concatenate((y_true1,y_true2,y_true3,y_true4,y_true5))
        print("Itr=%05d, trloss=%.5f, teloss=%.5f, traccuracy=%.5f, teaccuracy=%.5f" % (i, trloss, teloss, tracc, teacc))
        CM[i/10,:,:] = confusion_matrix(Y_true,Y_pred,isaccuracy=True)
        
    if i%100==0:
        xlabels = ['Horse','Building','Flower','Fruit','Landscape']
        ylabels = ['Horse','Building','Flower','Fruit','Landscape']
        mkdir('NIC_result')
        title_name = 'Itr_' + str(i)
        path = './NIC_result/'
        name = 'result'+str(i).zfill(5)+'.png'
        confusion_matrix_heatmap_pd(confusion_matrix(Y_true,Y_pred,isaccuracy=True), xlabels, ylabels, title_name, path, name)

np.save("./NIC_result/confusion_matrix.npy",CM)

























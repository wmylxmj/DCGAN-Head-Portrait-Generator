# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:02:16 2018

@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import time
import math

class DCGAN(object):
    '''
    Arguments:
        dataset_floder: the path of dataset
        image_height: output images' height
        image_width: output images' width
        batch_size: mini batch size
        max_trainset_size: max size of trainset
        z_dim: the len of z
        generator_filters_list: number of filters in generator's layers
        generator_kernel_size_list: kernel size of filters in generator's layers
        generator_kernel_strides_list: strides in conv2d
        discriminator_filters_list: number of filters in discriminator's layers
        discriminator_kernel_size_list: kernel size of filters in discriminator's layers
        discriminator_strides_list: strides in conv2d
        learning_rate: learning rate
        leaky_relu_alpha: leaky relu alpha
        adam_beta1: adam beta1
        epoch: num of epoches
        name: A string, name of the DCGAN
        images_save_floder: images save path
        n_plot_height: number of images in the height of plot images
        n_plot_width: number of images in the width of plot images
        plot_images_name: plot images name
    '''
    
    def __init__(self,
                 dataset_floder,
                 image_height=64,
                 image_width=64,
                 batch_size=64,
                 max_trainset_size=1024,       
                 z_dim=96,       
                 generator_filters_list=[1024, 512, 256, 128],
                 generator_kernel_size_list=[[5,5], [5,5], [5,5], [5,5]],
                 generator_kernel_strides_list=[(2,2), (2,2), (2,2), (2,2)],
                 discriminator_filters_list=[64, 128, 256, 512],
                 discriminator_kernel_size_list=[[5,5], [5,5], [5,5], [5,5]],
                 discriminator_strides_list=[(2,2), (2,2), (2,2), (2,2)],
                 learning_rate=0.0002,
                 leaky_relu_alpha=0.2,
                 adam_beta1=0.5,
                 epoch=500,
                 name=None,
                 images_save_floder='./results',
                 n_plot_height=None,
                 n_plot_width=None,
                 plot_images_name="DCGAN_GEN_"):
        assert(len(generator_filters_list)==len(generator_kernel_size_list) and \
               len(generator_filters_list)==len(generator_kernel_strides_list))
        assert(len(discriminator_filters_list)==len(discriminator_kernel_size_list) and \
               len(discriminator_filters_list)==len(discriminator_strides_list))
        self.dataset_floder = dataset_floder
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.max_trainset_size = max_trainset_size
        self.z_dim = z_dim
        self.generator_filters_list = generator_filters_list
        self.generator_kernel_size_list = generator_kernel_size_list
        self.generator_kernel_strides_list = generator_kernel_strides_list
        self.discriminator_filters_list = discriminator_filters_list
        self.discriminator_kernel_size_list = discriminator_kernel_size_list
        self.discriminator_strides_list = discriminator_strides_list
        self.learning_rate = learning_rate
        self.leaky_relu_alpha = leaky_relu_alpha
        self.adam_beta1 = adam_beta1
        self.epoch = epoch
        self.name = name
        self.images_save_floder = images_save_floder
        self.n_plot_height = n_plot_height
        self.n_plot_width = n_plot_width
        self.plot_images_name = plot_images_name
        self.make_dir(images_save_floder)
        pass
    
    def load_trainset(self):
        '''load trainset from the floder'''
        floder = self.dataset_floder
        # names of your train images
        images = os.listdir(floder)
        # the number of train images
        num_images = min(len(images), self.max_trainset_size)
        resize_height = self.image_height
        resize_width = self.image_width
        dataset = np.empty((num_images, resize_height, resize_width, 3), dtype="float32")
        for i in range(num_images):
            img = Image.open(floder + "/" + images[i])
            # size (w, h)
            # shape (h, w)
            img = img.resize((resize_width, resize_height)) 
            img_arr = np.asarray(img, dtype="float32") 
            dataset[i, :, :, :] = img_arr   
            pass
        with tf.Session() as sess: 
            sess.run(tf.initialize_all_variables())
            dataset = tf.reshape(dataset, [-1, resize_height, resize_width, 3])
            # range to -1, 1
            traindata = dataset * 1.0 / 127.5 - 1.0 
            # one vector
            traindata = tf.reshape(traindata, [-1, resize_height*resize_width*3])
            trainset = sess.run(traindata)
        print('[OK] ' + str(num_images) + ' samples have been loaded')
        return trainset
        
    def generator(self, z, reuse, trainable=True):
        '''creat a generator'''
        h_init = self.image_height
        w_init = self.image_width
        for i in range(len(self.generator_kernel_strides_list)):
            (h_stride, w_stride) = self.generator_kernel_strides_list[i]
            h_init = h_init / h_stride
            w_init = w_init / w_stride
            pass
        h_init = int(h_init)
        w_init = int(w_init)
        with tf.variable_scope("generator", reuse=reuse):
            # layer 1: FC
            output = tf.layers.dense(z, self.generator_filters_list[0]*h_init*w_init, \
                                     trainable=trainable)
            output = tf.reshape(output, [self.batch_size, h_init, w_init, \
                                         self.generator_filters_list[0]])
            output = tf.layers.batch_normalization(output, training=trainable)
            output = tf.nn.relu(output)
            # layer 2 to L-1
            for i in range(1, len(self.generator_filters_list)):
                filters = self.generator_filters_list[i]
                kernel_size = self.generator_kernel_size_list[i-1]
                strides = self.generator_kernel_strides_list[i-1]
                output = tf.layers.conv2d_transpose(output, filters, kernel_size, strides=strides, \
                                                    padding="SAME", trainable=trainable)
                output = tf.layers.batch_normalization(output, training=trainable)
                output = tf.nn.relu(output)
                pass
            # layer L
            kernel_size = self.generator_kernel_size_list[-1]
            strides = self.generator_kernel_strides_list[-1]
            output = tf.layers.conv2d_transpose(output, 3, kernel_size, strides=strides, \
                                                padding="SAME", trainable=trainable)
            generator_images = tf.nn.tanh(output)
        return generator_images
            
    def discriminator(self, x, reuse, trainable=True):
        '''creat a discriminator'''
        output = x
        with tf.variable_scope("discriminator", reuse=reuse):
            for i in range(len(self.discriminator_filters_list)):
                filters = self.discriminator_filters_list[i]
                kernel_size = self.discriminator_kernel_size_list[i]
                strides = self.discriminator_strides_list[i]
                output = tf.layers.conv2d(x, filters, kernel_size, strides=strides, \
                                          padding="SAME", trainable=trainable)
                output = tf.layers.batch_normalization(output, training=trainable)
                output = tf.nn.leaky_relu(output, alpha=self.leaky_relu_alpha)
                pass
            output = tf.layers.flatten(output)
            discriminator_output = tf.layers.dense(output, 1, trainable=trainable)
        return discriminator_output
    
    def make_dir(self, path):
        '''make the path'''
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
            print("You created a new path!")
            print("Path: " + str(path))
            pass
        else:
            print("Path: " + str(path) + " is already existed.")
        pass
    
    def plot_save_outputs(self, index, images):
        '''plot and save images'''
        n_height = self.n_plot_height
        n_width = self.n_plot_width
        batch_size = self.batch_size
        h = 0
        w = 0
        if n_height==None and n_width==None:           
            h = np.int(np.sqrt(batch_size))
            w = np.int(np.sqrt(batch_size))
        elif n_height!=None and n_width==None:
            h = n_height
            w = np.int(1.0*batch_size/h)
        elif n_height==None and n_width!=None:
            w = n_width
            h = np.int(1.0*batch_size/w)
        elif n_height!=None and n_width!=None:
            h = n_height
            w = n_width
            pass
        image_height = np.shape(images)[1]
        image_width = np.shape(images)[2]
        n_channel = np.shape(images)[3]
        images = np.reshape(images, [-1, image_height, image_width, n_channel])
        canvas = np.empty((h * image_height, w * image_width, n_channel))
        for i in range(h):
            for j in range(w):
                canvas[i*image_height:(i+1)*image_height, j*image_width:(j+1)*image_width, :] = \
                images[h*i+j].reshape(image_height, image_width, 3)
                pass
            pass
        plt.figure(figsize=(h, w))
        plt.imshow(canvas)
        label = "Epoch: {0}".format(index+1)
        plt.xlabel(label)
        file_name = self.plot_images_name + str(index+1)
        plt.savefig(self.images_save_floder + '/' + file_name)
        print(os.getcwd())
        print("[OK] image saved in file: ", self.images_save_floder + '/' + file_name)
        plt.show()
        pass
    
    def train(self):
        '''train'''
        trainset = self.load_trainset()
        x = tf.placeholder(tf.float32, shape=[None, self.image_height*self.image_width*3], name="input_real")
        X = tf.reshape(x, [-1] + [self.image_height, self.image_width, 3])
        z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="z")
        G = self.generator(z, reuse=False)
        D_fake_logits = self.discriminator(G, reuse=False)
        D_true_logits = self.discriminator(X, reuse=True)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, \
                                                                        labels=tf.ones_like(D_fake_logits)))
        D_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_true_logits, \
                                                                          labels=tf.ones_like(D_true_logits)))
        D_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, \
                                                                          labels=tf.zeros_like(D_fake_logits)))
        D_loss = D_loss_1 + D_loss_2
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_optimization = tf.train.AdamOptimizer(learning_rate=self.learning_rate, \
                                                    beta1=self.adam_beta1).minimize(G_loss, var_list=g_vars)
            d_optimization = tf.train.AdamOptimizer(learning_rate=self.learning_rate, \
                                                    beta1=self.adam_beta1).minimize(D_loss, var_list=d_vars)
        print("[OK] successfully make the network")
        start_time = time.time()   
        sess = tf.Session()
        sess.run(tf.initialize_all_variables()) 
        for i in range(self.epoch):
            total_batch = int(len(trainset)/self.batch_size)
            d_cost = 0
            g_cost = 0
            for j in range(total_batch):
                mini_batch = trainset[j*self.batch_size : (j*self.batch_size+self.batch_size)]
                z1 = np.random.uniform(low=-1.0, high=1.0, size=[self.batch_size, self.z_dim])
                d_op, d_loss = sess.run([d_optimization, D_loss], feed_dict={x: mini_batch, z: z1})
                z2 = np.random.uniform(low=-1.0, high=1.0, size=[self.batch_size, self.z_dim])
                g_op, g_loss = sess.run([g_optimization, G_loss], feed_dict={x: mini_batch, z: z2})
                images_generated = sess.run(G, feed_dict={z: z2})
                d_cost += d_loss/total_batch
                g_cost += g_loss/total_batch
                self.plot_save_outputs(i, images_generated)
                hour = int((time.time() - start_time)/3600)
                minute = int(((time.time() - start_time) - 3600*hour)/60)
                sec = int((time.time() - start_time) - 3600*hour - 60*minute)
                print("Time: ", hour, "h", minute, "m", sec, "s", \
                      "\nEpoch: ", i, "\nG_loss: ", g_cost, "\nD_loss: ", d_cost)
                pass
            pass
        sess.close()
        pass
    
    pass
    

tf.reset_default_graph()  

dcgan = DCGAN(dataset_floder='./faces', images_save_floder='./results_3') 
dcgan.train() 

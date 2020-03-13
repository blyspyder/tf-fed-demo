import tensorflow as tf
import numpy as np
import time
import random
import pickle
import math
import datetime
from keras.preprocessing.image import ImageDataGenerator

#预先定义的变量
class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9


#初始化权重，采用正则化随机初始，加入少量的噪声来打破对称性以及避免0梯度
def weight_variable(name, sp):
    initial = tf.initializers.he_normal()
    return tf.get_variable(name = name, shape = sp, initializer = initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                         updates_collections=None)
def conv(name,x,w,b):
    #去掉BN
    #return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME'),b),name=name)
    return tf.nn.relu(batch_norm(tf.nn.bias_add(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME'),b)),name=name)

def max_pool(name,x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)

def fc(name,x,w,b):
    return tf.nn.relu(batch_norm(tf.matmul(x,w)+b),name=name)

weights={
    'wc1_1' : weight_variable('wc1_1', [3,3,3,64]),
    'wc1_2' : weight_variable('wc1_2', [3,3,64,64]),
    'wc2_1' : weight_variable('wc2_1', [3,3,64,128]),
    'wc2_2' : weight_variable('wc2_2', [3,3,128,128]),
    'wc3_1' : weight_variable('wc3_1', [3,3,128,256]),
    'wc3_2' : weight_variable('wc3_2', [3,3,256,256]),
    'wc3_3' : weight_variable('wc3_3', [3,3,256,256]),
    'wc4_1' : weight_variable('wc4_1', [3,3,256,512]),
    'wc4_2' : weight_variable('wc4_2', [3,3,512,512]),
    'wc4_3' : weight_variable('wc4_3', [3,3,512,512]),
    'wc5_1' : weight_variable('wc5_1', [3,3,512,512]),
    'wc5_2' : weight_variable('wc5_2', [3,3,512,512]),
    'wc5_3' : weight_variable('wc5_3', [3,3,512,512]),
    'fc1' : weight_variable('fc1', [2*2*512,4096]),
    'fc2' : weight_variable('fc2', [4096,4096]),
    'fc3' : weight_variable('fc3', [4096,10])
}

biases={
    'bc1_1' : bias_variable([64]),
    'bc1_2' : bias_variable([64]),
    'bc2_1' : bias_variable([128]),
    'bc2_2' : bias_variable([128]),
    'bc3_1' : bias_variable([256]),
    'bc3_2' : bias_variable([256]),
    'bc3_3' : bias_variable([256]),
    'bc4_1' : bias_variable([512]),
    'bc4_2' : bias_variable([512]),
    'bc4_3' : bias_variable([512]),
    'bc5_1' : bias_variable([512]),
    'bc5_2' : bias_variable([512]),
    'bc5_3' : bias_variable([512]),
    'fb1' : bias_variable([4096]),
    'fb2' : bias_variable([4096]),
    'fb3' : bias_variable([10]),
}

#VGG-16网络，因为输入尺寸小，去掉最后两个个max pooling层
def vgg_net(input_shape,num_classes,learning_rate,graph):
    with graph.as_default():
        x = tf.placeholder(tf.float32,input_shape,name='X')
        y_ = tf.placeholder(tf.float32, [None, num_classes],name='Y')
        DROP_RATE = tf.placeholder(tf.float32, name='drop_rate')

        conv1_1=conv('conv1_1',x,weights['wc1_1'],biases['bc1_1'])
        conv1_2=conv('conv1_2',conv1_1,weights['wc1_2'],biases['bc1_2'])
        pool1=max_pool('pool1',conv1_2,k=2)

        conv2_1=conv('conv2_1',pool1,weights['wc2_1'],biases['bc2_1'])
        conv2_2=conv('conv2_2',conv2_1,weights['wc2_2'],biases['bc2_2'])
        pool2=max_pool('pool2',conv2_2,k=2)

        conv3_1=conv('conv3_1',pool2,weights['wc3_1'],biases['bc3_1'])
        conv3_2=conv('conv3_2',conv3_1,weights['wc3_2'],biases['bc3_2'])
        conv3_3=conv('conv3_3',conv3_2,weights['wc3_3'],biases['bc3_3'])
        pool3=max_pool('pool3',conv3_3,k=2)

        conv4_1=conv('conv4_1',pool3,weights['wc4_1'],biases['bc4_1'])
        conv4_2=conv('conv4_2',conv4_1,weights['wc4_2'],biases['bc4_2'])
        conv4_3=conv('conv4_3',conv4_2,weights['wc4_3'],biases['bc4_3'])
        pool4=max_pool('pool4',conv4_3,k=2)

        conv5_1=conv('conv5_1',pool4,weights['wc5_1'],biases['bc5_1'])
        conv5_2=conv('conv5_2',conv5_1,weights['wc5_2'],biases['bc5_2'])
        conv5_3=conv('conv5_3',conv5_2,weights['wc5_3'],biases['bc5_3'])
        pool5=max_pool('pool5',conv5_3,k=1)

        _shape=pool5.get_shape()
        flatten=_shape[1].value*_shape[2].value*_shape[3].value
        pool5=tf.reshape(pool5,shape=[-1,flatten])
        fc1=fc('fc1',pool5,weights['fc1'],biases['fb1'])
        fc1=tf.nn.dropout(fc1,DROP_RATE)

        fc2=fc('fc2',fc1,weights['fc2'],biases['fb2'])
        fc2=tf.nn.dropout(fc2,DROP_RATE)

        output=fc('fc3',fc2,weights['fc3'],biases['fb3'])

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=output))

        optimizer =  tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        prediction = tf.nn.softmax(output)
        pred = tf.argmax(prediction,1)

        correct_pred = tf.equal(pred, tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(
                tf.cast(correct_pred, tf.float32))

        return x,y_,DROP_RATE,train_op,loss_op,accuracy




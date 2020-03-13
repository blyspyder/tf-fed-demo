import tensorflow as tf
import numpy as np
from collections import namedtuple
import math
from Model import AlexNet
from Dataset import Dataset
import random
from VGG import vgg_net

# FedModel 定义包含属性 x,y,drop_rate,train_op,loss_op,acc_op等属性
FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op')

#联邦模型客户端类
class Clients:
    def __init__(self, input_shape, num_classes, learning_rate, clients_num):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        # 创建alxnet网络
        net = AlexNet(input_shape, num_classes, learning_rate, self.graph)
        #net = vgg_net(input_shape, num_classes, learning_rate, self.graph)
        self.model = FedModel(*net)

        # 初始化
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        # 装载数据
        # 根据训练客户端数量划分数据集
        self.dataset = Dataset(tf.keras.datasets.cifar10.load_data,split=clients_num)

        #self.dataset = Dataset(tf.keras.datasets.mnist.load_data,split=clients_num)


    #测试模型准确率
    def run_test(self, num, save=False):
        with self.graph.as_default():
            batch_x, batch_y = self.dataset.test.next_batch(num)
            #替代计算图中的x,y等数据
            feed_dict = {
                self.model.X: batch_x,
                self.model.Y: batch_y,
                self.model.DROP_RATE: 0
            }
        return self.sess.run([self.model.acc_op, self.model.loss_op],
                             feed_dict=feed_dict)

    def train_epoch(self, cid, batch_size=256, dropout_rate=0.4):
        dataset = self.dataset.train[cid]

        with self.graph.as_default():
            for _ in range(math.ceil(dataset.size // batch_size)):
                batch_x, batch_y = dataset.next_batch(batch_size)
                batch_x = data_augmentation(batch_x) #做数据增强处理
                feed_dict = {
                    self.model.X: batch_x,
                    self.model.Y: batch_y,
                    self.model.DROP_RATE: dropout_rate
                }
                self.sess.run(self.model.train_op, feed_dict=feed_dict)

#返回计算图中所有可训练的变量值
    def get_client_vars(self):
        """ Return all of the variables list """
        with self.graph.as_default():
            client_vars = self.sess.run(tf.trainable_variables())
        return client_vars

    def set_global_vars(self, global_vars):
        with self.graph.as_default():
            all_vars = tf.trainable_variables()#获取所有可训练变量
            for variable, value in zip(all_vars, global_vars):
                variable.load(value, self.sess)#加载server端发送的var到模型上

#随机返回ratio比例的客户端并返回编号
    def choose_clients(self, ratio=1.0):
        client_num = self.get_clients_num()
        choose_num = math.floor(client_num * ratio)
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        #返回客户端的数量
        return len(self.dataset.train)

#数据增强
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch

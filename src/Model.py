import tensorflow as tf
import numpy as np

def AlexNet(input_shape, num_classes, learning_rate, graph):
    with graph.as_default():
        X = tf.placeholder(tf.float32, input_shape, name='X')
        Y = tf.placeholder(tf.float32, [None, num_classes], name='Y')
        DROP_RATE = tf.placeholder(tf.float32, name='drop_rate')

        #定义核函数
        conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
        conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
        conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
        conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

        conv1 = tf.nn.conv2d(X, conv1_filter, strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv1_bn = tf.layers.batch_normalization(conv1_pool)

        conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.relu(conv2)
        conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv2_bn = tf.layers.batch_normalization(conv2_pool)

        conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
        conv3 = tf.nn.relu(conv3)
        conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv3_bn = tf.layers.batch_normalization(conv3_pool)

        conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
        conv4 = tf.nn.relu(conv4)
        conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv4_bn = tf.layers.batch_normalization(conv4_pool)

        flat = tf.contrib.layers.flatten(conv4_bn)

        full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
        full1 = tf.nn.dropout(full1, keep_prob=0.7)
        full1 = tf.layers.batch_normalization(full1)

        full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
        full2 = tf.nn.dropout(full2, keep_prob=0.7)
        full2 = tf.layers.batch_normalization(full2)

        full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
        full3 = tf.nn.dropout(full3, keep_prob=0.7)
        full3 = tf.layers.batch_normalization(full3)

        full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
        full4 = tf.nn.dropout(full4, keep_prob=0.7)
        full4 = tf.layers.batch_normalization(full4)

        logits = tf.contrib.layers.fully_connected(inputs=full4, num_outputs=10, activation_fn=None)

        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                        labels=Y))

        optimizer =  tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        #评估模型
        prediction = tf.nn.softmax(logits)
        pred = tf.argmax(prediction, 1)

        #m模型准确率
        correct_pred = tf.equal(pred, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32))

        return X, Y, DROP_RATE, train_op, loss_op, accuracy



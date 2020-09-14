#coding=utf-8

import tensorflow as tf
from tensorflow.python.framework import ops
import os
import numpy as np
from deep_regression.base_model import BaseModel
from tensorflow.python.training.moving_averages import assign_moving_average


class RegressionModel(BaseModel):
    def __init__(self, config, input_size):
        super(RegressionModel, self).__init__(config=config)
        self.output_path = config['output_path']
        self.hidden_size = config['hidden_size']
        # self.keep_prob = config['keep_prob']
        self.input_size = input_size

        # self.x = tf.placeholder(shape=(None, self.input_size), dtype=tf.float32, name='input')
        # self.y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='label')
        self.build_model()
        self.init_saver()

    def build_model(self):
        '''
        初始化w，b
        :return:
        '''
        parameters = {}
        output = None
        # ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
        with tf.name_scope('feed_forward'):
            for i, hidden_size in enumerate(self.hidden_size):
                with tf.name_scope('feed_forward'+str(i)):
                    if i==0:
                        output = self.forward_network(i, self.inputs, self.input_size, hidden_size)
                    else:
                        output = self.forward_network(i, output, self.hidden_size[i-1], self.hidden_size[i])
        output_size = output.get_shape()[-1].value
        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.001, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        # 计算交叉熵损失
        self.loss = self.cal_loss() + self.config["l2_reg_lambda"] * self.l2_loss
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()
        return output

    def forward_network(self, i, data, pre_hidden_size, cur_hidden_size):
        '''
        构建神经网络图
        :return:
        '''
        w = tf.get_variable('w'+str(i), [pre_hidden_size, cur_hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        # b = tf.get_variable('b', [[cur_hidden_size]], initializer=tf.zeros_initializer())
        b = tf.Variable(tf.constant(0.001, shape=[cur_hidden_size]), name="b"+str(i))

        dropout = tf.matmul(data, w) + b
        dropout = self.batch_norm(dropout)
        # dropout = tf.nn.leaky_relu(dropout, alpha=0.5)
        dropout = tf.nn.relu(dropout)
        with tf.name_scope("dropout"):
            dropout = tf.nn.dropout(dropout, self.keep_prob)

        return dropout

    def batch_norm(self, x):
        '''
        批次归一化
        :param x: 输入数据
        :return:
        '''
        return tf.layers.batch_normalization(x, axis=-1)


    @staticmethod
    def _batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = tf.shape(x)[-1:]
            moving_mean = tf.get_variable('mean', params_shape,
                                          initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_variance = tf.get_variable('variance', params_shape,
                                              initializer=tf.ones_initializer,
                                              trainable=False)

            def mean_var_with_update():
                mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                              assign_moving_average(moving_variance, variance, decay)]):
                    return tf.identity(mean), tf.identity(variance)

            mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape,
                                       initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape,
                                        initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            return x


    def predict(self, data, parameters):
        '''

        :param parameters:
        :return:
        '''
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            dataset = tf.cast(tf.constant(data), tf.float32)
            result = self.forward_network(dataset, parameters)
            prediction = result.eval()
        return prediction

    def rmse(self, predictions, labels):
        """
        calculate cost between two data sets
        :param predictions: data set of predictions
        :param labels: data set of labels (real values)
        :return: percentage of correct predictions
        """

        prediction_size = predictions.shape[0]
        prediction_cost = np.sqrt(np.sum(np.square(labels - predictions)) / prediction_size)

        return prediction_cost

    def rmsle(self, predictions, labels):
        """
        calculate cost between two data sets
        :param predictions: data set of predictions
        :param labels: data set of labels (real values)
        :return: percentage of correct predictions
        """

        prediction_size = predictions.shape[0]
        prediction_cost = np.sqrt(np.sum(np.square(np.log(predictions + 1) - np.log(labels + 1))) / prediction_size)

        return prediction_cost

    def l2_regularizer(self, cost, l2_beta, parameters, n_layers):
        """
        Function to apply l2 regularization to the model
        :param cost: usual cost of the model
        :param l2_beta: beta value used for the normalization
        :param parameters: parameters from the model (used to get weights values)
        :param n_layers: number of layers of the model
        :return: cost updated
        """

        regularizer = 0
        for i in range(1, n_layers):
            regularizer += tf.nn.l2_loss(parameters['w%s' % i])

        cost = tf.reduce_mean(cost + l2_beta * regularizer)

        return cost







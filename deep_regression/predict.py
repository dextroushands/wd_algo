import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import tensorflow as tf
from deep_regression.model import RegressionModel

class Predict(object):
    def __init__(self, config, input_size):
        self.model = None
        self.output_path = config['output_path']
        self.config = config
        self.input_size = input_size
        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def load_graph(self):
        '''
        加载计算图
        :return:
        '''
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                          self.config["ckpt_model_path"]))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))

    def create_model(self):
        '''
        初始化模型
        :return:
        '''
        self.model = RegressionModel(self.config, self.input_size)


    def predict(self, data):
        '''
        预测数据
        :param data:
        :return:
        '''
        prediction = self.model.infer(self.sess, data)
        return prediction
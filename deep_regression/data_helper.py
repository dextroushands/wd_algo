import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class TrainData(object):
    def __init__(self, config):
        self._train_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config["train_data"])
        self._test_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config["test_data"])
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                         config["output_path"])
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

    def read_data(self):

        train_data = pd.read_csv(self._train_data_path)
        test_data = pd.read_csv(self._test_data_path)

        return train_data, test_data

    def pre_process_data(self, df):
        """
        Perform a number of pre process functions on the data set
        :param df: pandas data frame
        :return: processed data frame
        """

        # one-hot encode categorical values
        df = pd.get_dummies(df)

        return df

    def data_pre(self):



        # TRAIN_PATH = '../data/train_cleaned.csv'
        # TEST_PATH = '../data/test_cleaned.csv'

        train, test = self.read_data()

        # get the labels values
        # train_raw_labels = train['SalePrice'].to_frame().as_matrix()
        # train_raw_labels = train['SalePrice'].values
        train_raw_labels = train['label'].values
        train_raw_labels = np.log1p(train_raw_labels)

        # pre process data sets
        # train_pre = self.pre_process_data(train)
        # test_pre = self.pre_process_data(test)
        # id = test_pre["Id"].values
        # # drop unwanted columns
        # train_pre = train_pre.drop(['Id', 'SalePrice'], axis=1)
        # test_pre = test_pre.drop(['Id'], axis=1)
        train_pre = train.drop(['label'], axis=1)
        test_pre = test


        # align both data sets (by outer join), to make they have the same amount of features,
        # this is required because of the mismatched categorical values in train and test sets
        # train_pre, test_pre = train_pre.align(test_pre, join='outer', axis=1)

        # replace the nan values added by align for 0
        train_pre.replace(to_replace=np.nan, value=0, inplace=True)
        test_pre.replace(to_replace=np.nan, value=0, inplace=True)

        # train_pre = train_pre.as_matrix().astype(np.float)
        # test_pre = test_pre.as_matrix().astype(np.float)

        # scale values
        # standard_scaler = preprocessing.StandardScaler()
        # train_pre = standard_scaler.fit_transform(train_pre)
        # test_pre = standard_scaler.fit_transform(test_pre)

        X_train, X_valid, Y_train, Y_valid = train_test_split(train_pre, train_raw_labels, test_size=0.3,
                                                              random_state=1)

        # hyperparameters
        input_size = train_pre.shape[1]

        return np.array(X_train), np.array(X_valid), np.array(Y_train), np.array(Y_valid), input_size, test_pre, id

    def next_batch(self, x, y, batch_size):
        """
        生成batch数据集
        :param x: 输入
        :param y: 标签
        :param batch_size: 批量的大小
        :return:
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = np.array(x[start: end], dtype="int64")
            batch_y = np.array(y[start: end], dtype="float32")

            yield dict(x=batch_x, y=batch_y)
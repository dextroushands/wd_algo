#coding=utf8

import os
import pandas as pd
import tensorflow as tf

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

class TimeData(object):
    '''
    时间数据处理
    '''
    def __init__(self, config):
        self.data_path = config['data_path']
        self.output_path = config['output_path']
        self.window_size = config['window_size']
        self.lag_size = config['lag_size']
        self.label_name = config['label_name']

    def read_data(self, size=None):
        '''
        数据读取
        :param n_row:
        :return:
        '''
        df = pd.read_csv(self.data_path, nrows=size)
        # df = pd.read_csv(csv_path)
        # slice [start:stop:step], starting from index 5 take every 6th record.
        # df = df[5::6]

        date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
        return df, date_time

    def feature_engineer(self, data, date_time):
        '''
        特征工程
        :param data:
        :return:
        '''
        df = data
        wv = df['wv (m/s)']
        bad_wv = wv == -9999.0
        wv[bad_wv] = 0.0

        max_wv = df['max. wv (m/s)']
        bad_max_wv = max_wv == -9999.0
        max_wv[bad_max_wv] = 0.0

        # The above inplace edits are reflected in the DataFrame
        df['wv (m/s)'].min()

        wv = df.pop('wv (m/s)')
        max_wv = df.pop('max. wv (m/s)')

        # Convert to radians.
        wd_rad = df.pop('wd (deg)') * np.pi / 180

        # Calculate the wind x and y components.
        df['Wx'] = wv * np.cos(wd_rad)
        df['Wy'] = wv * np.sin(wd_rad)

        # Calculate the max wind x and y components.
        df['max Wx'] = max_wv * np.cos(wd_rad)
        df['max Wy'] = max_wv * np.sin(wd_rad)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        day = 24 * 60 * 60
        year = (365.2425) * day

        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

        return df

    def series_to_supervised(self, data, window=1, lag=1, dropnan=True):
        '''
        时序数据转为监督学习
        :param data:
        :param window:
        :param lag:
        :param dropnan:
        :return:
        '''
        # n_vars = 1 if type(data) is list else data.shape[1]
        # df = DataFrame(data)
        cols, names = list(), list()
        self.feature_names = data.columns.tolist()
        self.feature_names.remove(self.label_name)
        #input sequences [t-n, ..., t-1]
        for i in range(window, 0, -1):
            cols.append(data.shift(i))
            names += [('%s(t-%d)' % (col, i)) for col in data.columns]
        #current time_step
        cols.append(data)
        names += [('%s(t)' % (col)) for col in data.columns]
        # Target timestep (t=lag)
        cols.append(data.shift(-lag))
        names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
        # Put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # Drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        self.label_name = '%s(t+%d)' % (self.label_name, lag)
        return agg

    def remove_data(self, df, window, lag, feature_names):
        '''
        去除不用的数据
        '''
        # feature_names = df.columns.tolist()
        # feature_names.remove(label_name)
        columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in feature_names]
        for i in range(window, 0, -1):
            columns_to_drop += [('%s(t-%d)' % (col, i)) for col in feature_names]
        columns_to_drop += [('%s(t)' % col) for col in feature_names]
        df.drop(columns_to_drop, axis=1, inplace=True)

        label = df.pop(self.label_name)
        df.insert(len(df.columns), self.label_name, label)

        return df


    def split_data(self, data):
        '''
        划分数据集
        :param data:
        :return:
        '''
        df = data
        column_indices = {name: i for i, name in enumerate(df.columns)}

        n = len(df)
        train_df = df[0:int(n * 0.7)]
        val_df = df[int(n * 0.7):int(n * 0.9)]
        test_df = df[int(n * 0.9):]

        num_features = df.shape[1]

        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        return train_df, val_df, test_df


    def data_gen(self):
        '''
        得到预处理数据
        '''
        #读取数据
        df, date_time = self.read_data(size=100000)
        #特征工程
        df = self.feature_engineer(df, date_time)
        #时序数据转换
        df = self.series_to_supervised(df, self.window_size, self.lag_size)
        #去除不用数据
        df = self.remove_data(df, self.window_size, self.lag_size, self.feature_names)
        #划分数据集
        train_df, val_df, test_df = self.split_data(df)
        return train_df, val_df, test_df


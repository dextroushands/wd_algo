import pandas as pd
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from time_series.data_helper import TimeData
from time_series.model import DnnModel
import json

# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)
# print(csv_path)

config_path = 'time_series/config.json'
with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config_path), 'r') as fr:
    config = json.load(fr)
timeData = TimeData(config)
train_df, val_df, test_df = timeData.data_gen()
print(train_df.shape[1])
dnn = DnnModel(config, train_df.shape[1]-1)
# model = dnn.model_structure()
model = dnn.lstm_structure()
history, model = dnn.train_lstm(model, train_df, val_df)
#输出 plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

dnn.predict_lstm(model, test_df)
#画出真实数据和预测数据
plt.plot(dnn.prediction,label='prediction')
plt.plot(test_df.iloc[:, -1].values,label='true')
plt.legend()
plt.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(test_df.iloc[:, -1].values, dnn.prediction))
print('Test RMSE: %.3f' % rmse)



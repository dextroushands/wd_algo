import json
import os
from deep_regression.predict import Predict
import pandas as pd
import numpy as np
from sklearn import preprocessing
from deep_regression.data_helper import TrainData

config_path = 'deep_regression/config.json'
with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config_path), "r") as fr:
    config = json.load(fr)
train_data = TrainData(config)
_,_,_,_,_,test_pre,_= train_data.data_pre()
# test_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config['test_data'])
# test_df = pd.read_csv(test_path)
# id = test_pre["Id"].values
#
# test_pre = test_pre.drop(['Id'], axis=1)
# test_df = pd.get_dummies(test_df)
# test_df.replace(to_replace=np.nan, value=0, inplace=True)
# test_df = test_df.as_matrix().astype(np.float)

input_size = test_pre.shape[1]
# standard_scaler = preprocessing.StandardScaler()
# test_df = standard_scaler.fit_transform(test_df)

predictor = Predict(config, input_size)


result = predictor.predict(np.array(test_pre))
result = [(np.exp(res)-1)[0] for res in result]
print('predict results...')
print(result)
# submission = pd.DataFrame(data=None, columns=['Id', 'SalePrice'])
# submission['Id'] = id
# submission['SalePrice'] = result
# submission.to_csv('submission2.csv', index=0)
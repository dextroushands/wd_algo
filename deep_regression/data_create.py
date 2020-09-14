import numpy as np
import pandas as pd

def data_gen():
    '''
    生成随机数
    :return:
    '''
    param = [1,2,3,4,5]
    samples = sample_gen(100,30,5)
    samples = [int(i) for i in samples]
    # result = int(np.dot(param,samples))
    # samples = samples.tolist()
    # samples.append(result)

    return samples


def sample_gen(loc, scale, size):
    '''
    高斯分布随机数
    :param loc:均值
    :param scale:标准差
    :param size:随机数的个数
    :return:
    '''
    return np.random.normal(loc,scale,size)

def save_to_csv(file_path, n_row):
    '''
    将生成的数据保存到csv
    :param file_path: 路径
    :param n_row: 数据条数
    :return:
    '''
    # columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'label']
    columns = ['f1', 'f2', 'f3', 'f4', 'f5']

    data = []
    for i in range(n_row):
        data.append(data_gen())
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=0)

if __name__=='__main__':
    print(sample_gen(100,30,5))
    print(data_gen())
    file_path = '../data/example_test.csv'
    save_to_csv(file_path, 100)
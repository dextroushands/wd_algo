'''
定义各类性能指标
'''

from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, mean_squared_log_error


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res

def get_metrics(y_pred, y_true):
    '''
    计算各类性能指标
    :param y_true:
    :param y_pred:
    :return:
    '''
    var_score = explained_variance_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # msle = mean_squared_log_error(y_true, y_pred)

    return var_score, mse, mae
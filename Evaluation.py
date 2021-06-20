import numpy as np


def HR(array, thres=10):
    return np.sum(np.where(array <= thres, 1, 0)) / array.size


HR10 = lambda x: HR(x, 10)
HR20 = lambda x: HR(x, 20)


def NDCG(array):
    return np.sum(1 / np.log2(array + 1)) / array.size

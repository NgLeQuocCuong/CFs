from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
import numpy as np


class Dataset(object):
    def __init__(self, dataset, u_field='user_id', i_field='item_id', r_field='rating', t_field='timestampe'):
        self._dataset = read_csv(dataset).rename(columns={u_field: 'user_id', i_field: 'item_id', r_field: 'rating', t_field: 'timestamp'})[['user_id', 'item_id', 'rating', 'timestamp']]
        self._train = None
        self._test = None

    def filter_data(self, thres):
        result = self._dataset.copy()
        l = len(result)+1
        while(l - len(result)):
            l = len(result)
            result = result[result.groupby('user_id').user_id.transform(len) >= thres]
            result = result[result.groupby('item_id').item_id.transform(len) >= thres]
        self._dataset = result


    def prepare_input(self):
        self._dataset['u_cat'] = self._dataset.user_id.astype('category').cat.codes.values
        self._dataset['i_cat'] = self._dataset.item_id.astype('category').cat.codes.values
        n_u = len(self._dataset.u_cat.unique())
        n_i = len(self._dataset.i_cat.unique())
        def fn(group, n):
            dictionary = {}
            for i in group.to_list():
                dictionary[i] = True
            v = np.array([1 if _ in dictionary else 0 for _ in range(n)])
            return [v for _ in group.to_list()]
        prepare_user = lambda x: fn(x, n_i)
        prepare_item = lambda x: fn(x, n_u)
        self._dataset['user_data'] = self._dataset.groupby('u_cat').i_cat.transform(prepare_user)
        self._dataset['item_data'] = self._dataset.groupby('i_cat').u_cat.transform(prepare_item)
        self._dataset = self._dataset.drop(columns=['u_cat', 'i_cat'])
        
    def prepare_train_test(self, by_last_rate=True, test_rate=None):
        if test_rate:
            self._train, self._test = train_test_split(self._dataset, test_size=test_rate)
        elif by_last_rate:
            self._train = self._dataset[self._dataset.groupby('user_id').timestamp.transform(max) != self._dataset['timestamp']]
            self._test = self._dataset[self._dataset.groupby('user_id').timestamp.transform(max) == self._dataset['timestamp']]

    def get_trainset(self):
        return self._train

    def get_testset(self):
        return self._test

    def get_dataset(self):
        return self._dataset
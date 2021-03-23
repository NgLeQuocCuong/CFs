from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import Tensor


class Dataset(object):
    def __init__(self, dataset, u_field='user_id', i_field='item_id', r_field='rating', t_field='timestampe'):
        self.dataset = read_csv(dataset).rename(columns={u_field: 'user_id', i_field: 'item_id', r_field: 'rating', t_field: 'timestamp'})[['user_id', 'item_id', 'rating', 'timestamp']]
        self._train = None
        self._test = None

    def filter_data(self, thres):
        result = self.dataset.copy()
        l = len(result)+1
        while(l - len(result)):
            l = len(result)
            result = result[result.groupby('user_id').user_id.transform(len) >= thres]
            result = result[result.groupby('item_id').item_id.transform(len) >= thres]
        self.dataset = result


    def prepare_input(self):
        self.dataset['u_cat'] = self.dataset.user_id.astype('category').cat.codes.values
        self.dataset['i_cat'] = self.dataset.item_id.astype('category').cat.codes.values
        n_u = len(self.dataset.u_cat.unique())
        n_i = len(self.dataset.i_cat.unique())
        user_size = self.dataset.groupby('user_id').user_id.transform(len).max()
        item_size = self.dataset.groupby('item_id').item_id.transform(len).max()
        def fn(group, size):
            v = group.to_list() + [-1 for _ in range(size-len(group.to_list()))]
            return [v for _ in group.to_list()]
        prepare_user = lambda x: fn(x, user_size)
        prepare_item = lambda x: fn(x, item_size)
        user_data = self.dataset.groupby('u_cat').i_cat.transform(prepare_user)
        item_data = self.dataset.groupby('i_cat').u_cat.transform(prepare_item)
        self.dataset = self.dataset.drop(columns=['u_cat', 'i_cat'])
        return [np.array(user_data.to_list()), np.array(item_data.to_list())]
        
    def prepare_train_test(self, by_last_rate=True, test_rate=None):
        if test_rate:
            self._train, self._test = train_test_split(self.dataset, test_size=test_rate)
        elif by_last_rate:
            self._train = self.dataset[self.dataset.groupby('user_id').timestamp.transform(max) != self.dataset['timestamp']]
            self._test = self.dataset[self.dataset.groupby('user_id').timestamp.transform(max) == self.dataset['timestamp']]

    def get_trainset(self):
        return self._train

    def get_testset(self):
        return self._test

    def get_dataset(self):
        return self.dataset

    def get_train_label(self):
        return self._train['rating'].to_numpy()
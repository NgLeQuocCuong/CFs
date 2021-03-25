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


    def prepare_user_input(self):
        self._train['i_cat'] = self._train.item_id.astype('category').cat.codes.values
        n_i = len(self._train.i_cat.unique())
        data = self._train[['user_id']].drop_duplicates(subset=['user_id'], ignore_index=True)
        def prepare(uid, df, n):
          lst = df[df['user_id'] == uid].i_cat.to_list()
          return np.array([1 if _ in lst else 0 for _ in range(n)])
        data['data'] = data.user_id.apply(prepare, args=(self._train, n_i))
        self.user_input = data
        self._train = self._train.drop(columns = ['i_cat'])

    def prepare_item_input(self):
        self._train['u_cat'] = self._train.user_id.astype('category').cat.codes.values
        n_u = len(self._train.u_cat.unique())
        data = self._train[['item_id']].drop_duplicates(subset=['item_id'], ignore_index=True)
        def prepare(uid, df, n):
          lst = df[df['item_id'] == uid].u_cat.to_list()
          return np.array([1 if _ in lst else 0 for _ in range(n)])
        data['data'] = data.item_id.apply(prepare, args=(self._train, n_u))
        self.item_input = data
        self._train = self._train.drop(columns = ['u_cat'])


    def prepare_train_test(self, by_last_rate=True, test_rate=None):
        if test_rate:
            self._train, self._test = train_test_split(self.dataset, test_size=test_rate)
        elif by_last_rate:
            self._train = self.dataset[self.dataset.groupby('user_id').timestamp.transform(max) != self.dataset['timestamp']]
            self._test = self.dataset[self.dataset.groupby('user_id').timestamp.transform(max) == self.dataset['timestamp']]
        self.dataset = None

    def get_train_data(self, rate=1):
        if rate == 1:
            data = self._train
        else:
            _, data = train_test_split(self._train, test_size=rate)
        del _
        data = data.merge(self.user_input, left_on='user_id', right_on='user_id')
        data = data.merge(self.item_input, left_on='item_id', right_on='item_id', suffixes=('_user', '_item'))
        return data['data_user'], data['data_item'], data['rating']
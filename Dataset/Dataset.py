from pandas import read_csv
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, dataset, u_field='user_id', i_field='item_id', r_field='rating', t_field='timestampe'):
        self._dataset = read_csv(dataset).rename(columns={u_field: 'user_id', i_field: 'item_id', r_field: 'rating', t_field: 'timestamp'})
        self._train = None
        self._test = None
        self._dataset.user_cat = self._dataset.user_id.astype('category').cat.codes.values
        self._dataset.item_cat = self._dataset.item_id.astype('category').cat.codes.values

    def filter_data(self, thres):
        result = self._dataset.copy()
        l = len(result)+1
        while(l - len(result)):
            l = len(result)
            result = result[result.groupby('user_id').user_id.transform(len) >= thres]
            result = result[result.groupby('item_id').item_id.transform(len) >= thres]
        self._dataset = result
        self._dataset.user_cat = self._dataset.user_id.astype('category').cat.codes.values
        self._dataset.item_cat = self._dataset.item_id.astype('category').cat.codes.values

    def prepare_train_test(self, by_last_rate=True, test_rate=None):
        if test_rate:
            self._train, self._test = train_test_split(self._dataset, test_size=test_rate)
        else if by_last_rate:
            self._train = self._dataset[self._dataset.groupby('user_id').timestamp.transform(max) != self._dataset['timestamp']]
            self._test = self._dataset[self._dataset.groupby('user_id').timestamp.transform(max) == self._dataset['timestamp']]

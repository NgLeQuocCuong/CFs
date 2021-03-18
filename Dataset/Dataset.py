from pandas import read_csv, DataFrame
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


    def prepare_input(self):
        self._user = DataFrame(self._dataset['user_id'])
        self._item = DataFrame(self._dataset['item_id'])
        n_u = len(self._dataset.u_cat.unique())
        n_i = len(self._dataset.i_cat.unique())
        def fn(group, n):
            dictionary = {}
            for i in group.to_list():
                dictionary[i] = True
            v = [1 if _ in d else 0 for _ in range(n)]
            return [v for _ in x.to_numpy()]
        prepare_user = lambda x: fn(x, n_i)
        prepare_item = lambda x: fn(x, n_u)
        self._user['data'] = df.groupby('u_cat').i_cat.transform(prepare_user)
        self._item['data'] = df.groupby('i_cat').u_cat.transform(prepare_item)
        self._user = self._user.drop_duplicates(subset=['user_id'], ignore_index=True)
        self._item = self._item.drop_duplicates(subset=['item_id'], ignore_index=True)
        
    def prepare_train_test(self, by_last_rate=True, test_rate=None):
        if test_rate:
            self._train, self._test = train_test_split(self._dataset, test_size=test_rate)
        elif by_last_rate:
            self._train = self._dataset[self._dataset.groupby('user_id').timestamp.transform(max) != self._dataset['timestamp']]
            self._test = self._dataset[self._dataset.groupby('user_id').timestamp.transform(max) == self._dataset['timestamp']]

    def get_users(self):
        return self._user

    def get_items(self):
        return self._item
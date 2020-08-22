import pickle
import pandas as pd
import numpy as np

class DatasetGen:
    def __init__(self, log_file='data/data_format1/user_log_format1.csv',
                 user_info_file='data/data_format1/user_info_format1.csv',
                 dataset_file='data/tmall_dataset.pkl',
                 appear_time=10,
                 min_length=3):
        df = pd.read_csv(log_file)
        df.drop_duplicates(inplace=True)
        self.df = df[np.logical_or(df.action_type == 1, df.action_type == 2)]
        self.df.sort_values(by=['user_id', 'time_stamp'], inplace=True)
        self.item_buy(appear_time)
        self.user_info_df = pd.read_csv(user_info_file)
        self.process_user_id()
        self.reset_id('user_id')
        self.reset_id('item_id')
        self.reset_id('cat_id')
        self.reset_id('seller_id')
        self.reset_id('brand_id')
        self.dataset_file = dataset_file
        self.data_set = []
        self.generate_data_set(min_length)
        self.data_set.sort(key=lambda x:len(x))
        with open(dataset_file, 'wb') as f:
            pickle.dump((self.data_set,
                         self.user_info), f)

    def item_buy(self, appear_time=10):
        count = self.df.groupby(by=['item_id']).item_id.agg('count')
        items = set(count[count >= appear_time].index)
        self.df = self.df[self.df.item_id.isin(items)]

    def process_user_id(self):
        map_dict = {id: i for i, id in enumerate(set(self.df.user_id.values))}
        self.df.user_id = self.df.user_id.apply(lambda x: map_dict[x])
        self.user_info_df = self.user_info_df[self.user_info_df.user_id.isin(map_dict.keys())]
        self.user_info_df['user_id'] = self.user_info_df['user_id'].apply(lambda x: map_dict[x])
        self.user_info = [(row.age_range, row.gender) for i, row in self.user_info_df.iterrows()]

    def reset_id(self, column_name):
        map_dict = {id: i for i, id in enumerate(set(self.df[column_name].values))}
        self.df[column_name] = self.df[column_name].apply(lambda x: map_dict[x])

    def generate_data_set(self, min_length=3):
        for i, group in self.df.groupby('user_id'):
            if len(group) < min_length or len(set(group.time_stamp.values)) < 2:
                continue
            self.data_set.append(group[:-1])

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
debug = True
class DataFetch:
    def __init__(self, dataset_path='data/tmall_dataset.pkl', portion=2/3, device=torch.device('cpu')):
        self.device = device
        self.portion = portion
        with open(dataset_path, 'rb') as f:
            self.data_set, self.user_info = pickle.load(f)

    def process_variable_length_input(self, ids):
        # type:(list[np.ndarray])->object
        sequences = [torch.from_numpy(item).long().to(self.device) for item in ids]
        lengths = [len(item) for item in ids]
        sequences = pad_sequence(sequences, batch_first=True,padding_value=0)
        return lengths, sequences

    def gen_mask(self, lengths):
        return pad_sequence([torch.ones(n, dtype=torch.float32).long().to(self.device)
                             for n in lengths], batch_first=True, padding_value=0)


    def batch2tensor(self, batch, train=True):
        user_ids = batch.user_id.values
        age_ranges = []
        genders = []
        for user_id in user_ids:
            age_range, gender = self.user_info[user_id]
            age_ranges.append(age_range)
            genders.append(gender)

        item_ids = []
        cat_ids = []
        seller_ids = []
        brand_ids = []
        # timestamps = []
        # action_types = []
        for group in batch:
            k = round(len(group) * self.portion)
            if k == len(group):
                k -= 1
            block = group.iloc[:k] if train else group.iloc[k:]
            item_ids.append(block.item_id.values)
            cat_ids.append(block.cat_id.values)
            seller_ids.append(block.seller_id.values)
            brand_ids.append(block.brand_id.values)
            # timestamps.append(block.time_stamp.values)
            # action_types.append(block.action_type.values)
            if debug:
                assert set(block.use_id) == 1

        user_ids = torch.from_numpy(np.array(user_ids)).long().to(self.device)
        age_ranges = F.one_hot(age_ranges, n=9).float().to(self.device)
        genders = F.one_hot(genders, n=3).float().to(self.device)

        lengths, item_ids = self.process_variable_length_input(item_ids)
        _, cat_ids = self.process_variable_length_input(cat_ids)
        _, seller_ids = self.process_variable_length_input(seller_ids)
        _, brand_ids = self.process_variable_length_input(brand_ids)
        return user_ids, age_ranges, genders, item_ids, cat_ids, seller_ids, brand_ids, self.gen_mask(lengths), lengths

    def training_set(self, batch_size=128):
        idx = 0
        n = len(self.data_set)
        while idx < n:
            yield self.batch2tensor(self.data_set[idx:min(idx + batch_size, n)], True)
            idx += batch_size

    def testing_set(self, batch_size=128):
        idx = 0
        n = len(self.data_set)
        while idx < n:
            yield self.batch2tensor(self.data_set[idx:min(idx + batch_size, n)], False)
            idx += batch_size


if __name__ == '__main__':
    DatasetGen()

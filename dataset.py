from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch


class Mydata(Dataset):
    def __init__(self, S_path, T_path, test_path, train_ratio=0.9, val_ratio=0.1,
                 train=True, preprocessed=True):
        super().__init__()

        self.S_path = S_path
        self.T_path = T_path
        self.test_path = test_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.train = train
        if not preprocessed:
            self.S_df = pd.read_csv(S_path, header=None)
            self.T_df = pd.read_csv(T_path, header=None)
            self.test_df = pd.read_csv(test_path, header=None)
            self.S_df.columns = ["user", "item", "label"]
            self.T_df.columns = ["user", "item", "label"]
            self.test_df.columns = ["user", "item", "label"]

            S_item_cnt = Counter(self.S_df.iloc[:, 1])
            S_item_cnt = {k: v for k, v in S_item_cnt.items() if v >= 5}
            T_item_cnt = Counter(self.T_df.iloc[:, 1])
            T_item_cnt = {k: v for k, v in T_item_cnt.items() if v >= 5}
            # Find the intersection of items
            S_item = set(S_item_cnt.keys())
            T_item = set(T_item_cnt.keys())
            items = list(S_item.intersection(T_item))
            self.S_df = self.S_df.loc[self.S_df["item"].isin(items)]
            self.T_df = self.T_df.loc[self.T_df["item"].isin(items)]
            self.test_df = self.test_df.loc[self.test_df["item"].isin(items)]
            self.test_df = self.test_df.loc[self.test_df["user"].isin(self.T_df["user"])]
            # All items have a unified id
            dict_items = {items[i]: i for i in range(len(items))}

            S_users = list(set(self.S_df.iloc[:, 0]))
            T_users = list(set(self.T_df.iloc[:, 0]))
            # All users have a unified id
            dict_S_users = {S_users[i]: i for i in range(len(S_users))}
            dict_T_users = {T_users[i]: i for i in range(len(T_users))}
            self.S_df.reset_index(drop=True, inplace=True)
            self.T_df.reset_index(drop=True, inplace=True)
            self.test_df.reset_index(drop=True, inplace=True)
            # Change code to id
            for index, row in tqdm(self.S_df.iterrows()):
                item_idx = dict_items[row["item"]]
                user_idx = dict_S_users[row["user"]]
                self.S_df.iloc[index, 0] = user_idx
                self.S_df.iloc[index, 1] = item_idx
            for index, row in tqdm(self.T_df.iterrows()):
                item_idx = dict_items[row["item"]]
                user_idx = dict_T_users[row["user"]]
                self.T_df.iloc[index, 0] = user_idx
                self.T_df.iloc[index, 1] = item_idx
            for index, row in tqdm(self.test_df.iterrows()):
                item_idx = dict_items[row["item"]]
                user_idx = dict_T_users[row["user"]]
                self.test_df.iloc[index, 0] = user_idx
                self.test_df.iloc[index, 1] = item_idx
            # Save processed csv for calculating metrics
            self.S_df.to_csv(self.S_path + ".processed.csv", index=False)
            self.T_df.to_csv(self.T_path + ".processed.csv", index=False)
            self.test_df.to_csv(self.test_path + ".processed.csv", index=False)
            # 样例
            # Item    User    Rating
            # 22      32          5
            print(len(items))
            # 构建rating matrix
            self.S_data = torch.zeros((len(items), len(S_users)))
            self.T_data = torch.zeros((len(items), len(T_users)))
            for index, row in tqdm(self.S_df.iterrows()):
                user = row["user"]
                item = row["item"]
                self.S_data[item, user] = int(row["label"])

            for index, row in tqdm(self.T_df.iterrows()):
                user = row["user"]
                item = row["item"]
                self.T_data[item, user] = int(row["label"])

            np.save(self.S_path + '.npy', self.S_data)
            np.save(self.T_path + '.npy', self.T_data)

        else:
            self.S_data = np.load(self.S_path + '.npy')
            self.T_data = np.load(self.T_path + '.npy')
            self.S_df = pd.read_csv(self.S_path + ".processed.csv")
            self.T_df = pd.read_csv(self.T_path + ".processed.csv")
            self.test_df = pd.read_csv(self.test_path + ".processed.csv")
        np.random.seed(42)
        self.S_y = torch.zeros((self.S_data.shape[0], 1))
        self.T_y = torch.ones((self.T_data.shape[0], 1))
        self.total_indices = np.arange(self.S_data.shape[0]).astype(int)
        self.val_indices = np.random.choice(self.total_indices, size=int(len(self.S_data) * self.val_ratio),
                                            replace=False).astype(int)
        self.train_indices = np.array(list(set(self.total_indices) - set(self.val_indices))).astype(int)

        if train:
            self.S_data = self.S_data[self.train_indices]
            self.T_data = self.T_data[self.train_indices]
            self.S_y = self.S_y[self.train_indices]
            self.T_y = self.T_y[self.train_indices]
        else:
            self.S_data = self.S_data[self.val_indices]
            self.T_data = self.T_data[self.val_indices]
            self.S_y = self.S_y[self.val_indices]
            self.T_y = self.T_y[self.val_indices]

    def __getitem__(self, item):
        return self.S_data[item], self.T_data[item], self.S_y[item], self.T_y[item]

    def __len__(self):
        return self.S_data.shape[0]


if __name__ == "__main__":
    data = Mydata(r'dataset\data_source.csv',
                  r'dataset\data_target.csv',
                  r'dataset\data_test.csv', train=True, preprocessed=False)

    dataloader = DataLoader(data, batch_size=8, shuffle=False)
    for batch in dataloader:
        print(batch)
        break

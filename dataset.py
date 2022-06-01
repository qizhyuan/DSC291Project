from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch


class Mydata(Dataset):
    def __init__(self, S_path, T_path, test_path, preprocessed=True):
        super().__init__()

        self.S_path = S_path
        self.T_path = T_path
        self.test_path = test_path
        if not preprocessed:
            self.S_df = pd.read_csv(S_path)
            self.T_df = pd.read_csv(T_path)
            self.test_df = pd.read_csv(test_path)
            self.S_df.columns = ["user", "item", "label"]
            self.T_df.columns = ["user", "item", "label"]
            self.test_df.columns = ["user", "item", "label"]
            # Find the intersection of items
            items = list(set(self.S_df["item"]))
            # All items have a unified id
            dict_items = {items[i]: i for i in range(len(items))}
            # All users have a unified id
            S_users = list(set(self.S_df.iloc[:, 0]))
            T_users = list(set(self.T_df.iloc[:, 0]))
            dict_S_users = {S_users[i]: i for i in range(len(S_users))}
            dict_T_users = {T_users[i]: i for i in range(len(T_users))}
            # Change code to id
            for index, row in tqdm(self.S_df.iterrows()):
                self.S_df.iloc[index, 0] = dict_S_users[row["user"]]
                self.S_df.iloc[index, 1] = dict_items[row["item"]]
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
            # 样例
            # Item    User    Rating
            # 22      32          5
            val_T_df = self.T_df.iloc[-370:, :]
            self.T_df = self.T_df.iloc[:-370, :]
            val_T_df.to_csv(self.T_path + ".val.processed.csv", index=False)
            self.S_df.to_csv(self.S_path + ".processed.csv", index=False)
            self.T_df.to_csv(self.T_path + ".processed.csv", index=False)
            self.test_df.to_csv(self.test_path + ".processed.csv", index=False)
            print(len(items))
            # 构建rating matrix
            self.S_data = torch.zeros((len(items), len(S_users)))
            self.T_data = torch.zeros((len(items), len(T_users)))
            for index, row in tqdm(self.S_df.iterrows()):
                user = int(row["user"])
                item = int(row["item"])
                self.S_data[item, user] = 1

            for index, row in tqdm(self.T_df.iterrows()):
                user = int(row["user"])
                item = int(row["item"])
                self.T_data[item, user] = 1

            np.save(self.S_path + '.npy', self.S_data)
            np.save(self.T_path + '.npy', self.T_data)
        else:
            self.S_data = np.load(self.S_path + '.npy')
            self.T_data = np.load(self.T_path + '.npy')
            self.S_df = pd.read_csv(self.S_path + ".processed.csv")
            self.T_df = pd.read_csv(self.T_path + ".processed.csv")
            self.test_df = pd.read_csv(self.test_path + ".processed.csv")
        self.S_y = torch.zeros((self.S_data.shape[0], 1))
        self.T_y = torch.ones((self.T_data.shape[0], 1))

    def __getitem__(self, item):
        return self.S_data[item], self.T_data[item], self.S_y[item], self.T_y[item]

    def __len__(self):
        return self.S_data.shape[0]


if __name__ == "__main__":
    data = Mydata(r'dataset\data_source.csv',
                  r'dataset\data_target.csv',
                  r'dataset\data_test.csv', preprocessed=False)

    dataloader = DataLoader(data, batch_size=8, shuffle=False)
    for batch in dataloader:
        print(batch)
        break

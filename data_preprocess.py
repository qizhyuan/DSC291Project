import pandas as pd
import numpy as np


source = pd.read_csv("dataset/data_source.csv.processed.csv")
target = pd.read_csv("dataset/data_target.csv.processed.csv")
test = pd.read_csv("dataset/data_test.csv.processed.csv")

source["label"] = 1
target["label"] = 1
test["label"] = 1

target_val = target.iloc[-370:, :]
target = target.iloc[:-370, :]
target_val.reset_index(drop=True, inplace=True)
target.reset_index(drop=True, inplace=True)
print(target_val.shape)
print(target.shape)

source.to_csv("dataset/data_source_processed.csv")
target.to_csv("dataset/data_target_processed.csv")
target_val.to_csv("dataset/data_target_val_processed.csv")
test.to_csv("dataset/data_test_processed.csv")

source_rating_matrix = np.load("dataset/data_source.csv.npy")
target_rating_matrix = np.load("dataset/data_target.csv.npy")
print(source_rating_matrix.shape)
print(target_rating_matrix.shape)

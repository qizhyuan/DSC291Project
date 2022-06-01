import numpy as np
import matplotlib.pyplot as plt
import json


loss = np.load("log/train_rmse.npy")
hits1 = np.load("log/val_hits1.npy")
hits5 = np.load("log/val_hits5.npy")
hits10 = np.load("log/val_hits10.npy")
hits15 = np.load("log/val_hits15.npy")
ndcg1 = np.load("log/val_ndcg1.npy")
ndcg5 = np.load("log/val_ndcg5.npy")
ndcg10 = np.load("log/val_ndcg10.npy")
ndcg15 = np.load("log/val_ndcg15.npy")

print(loss)

plt.plot(range(1, 51), hits1, label="hits1")
plt.plot(range(1, 51), hits5, label="hits5")
plt.plot(range(1, 51), hits10, label="hits10")
plt.plot(range(1, 51), hits15, label="hits15")
plt.plot(range(1, 51), ndcg1, label="NDCG1")
plt.plot(range(1, 51), ndcg5, label="NDCG5")
plt.plot(range(1, 51), ndcg10, label="NDCG10")
plt.plot(range(1, 51), ndcg15, label="NDCG15")
plt.legend()
plt.show()

plt.plot(range(1, 51), loss)
plt.show()





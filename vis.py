import numpy as np
import matplotlib.pyplot as plt
import json


S_ae_loss = np.load("log/autoencoder_S_loss.npy")
T_ae_loss = np.load("log/autoencoder_T_loss.npy")

plt.plot(range(1, 31), S_ae_loss, label="Source autoencoder")
plt.plot(range(1, 31), T_ae_loss, label="Target autoencoder")
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Losses of pretraining Autoencoders", fontsize=14)
plt.xticks(range(2, 31, 2), labels=[str(num) for num in range(2, 31, 2)])
plt.legend(loc="upper right", fontsize=14)
plt.savefig(r"images\autoencoder_loss.jpg", dpi=300, bbox_inches="tight")
plt.show()

loss = np.load("log/train_rmse.npy")
hits1 = np.load("log/val_hits1.npy")
hits5 = np.load("log/val_hits5.npy")
hits10 = np.load("log/val_hits10.npy")
hits15 = np.load("log/val_hits15.npy")
ndcg1 = np.load("log/val_ndcg1.npy")
ndcg5 = np.load("log/val_ndcg5.npy")
ndcg10 = np.load("log/val_ndcg10.npy")
ndcg15 = np.load("log/val_ndcg15.npy")


plt.plot(range(1, 51), hits1, label="hits1")
plt.plot(range(1, 51), hits5, label="hits5")
plt.plot(range(1, 51), hits10, label="hits10")
plt.plot(range(1, 51), hits15, label="hits15")
plt.plot(range(1, 51), ndcg1, label="NDCG1")
plt.plot(range(1, 51), ndcg5, label="NDCG5")
plt.plot(range(1, 51), ndcg10, label="NDCG10")
plt.plot(range(1, 51), ndcg15, label="NDCG15")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Metrics", fontsize=14)
plt.title("Metrics on validation set", fontsize=14)
plt.savefig(r"images\val_metrics.jpg", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(range(1, 51), loss)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Loss on training set", fontsize=14)
plt.savefig(r"images\train_loss.jpg", dpi=300, bbox_inches="tight")
plt.show()





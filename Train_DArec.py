import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data
from model import *
from torch.utils.data import DataLoader
from dataset import Mydata
import argparse
from tqdm import tqdm
import math
import json
from util import downsampling

parser = argparse.ArgumentParser(description='DArec with PyTorch')
parser.add_argument('--epochs', '-e', type=int, default=50)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-5)
parser.add_argument("--n_factors", type=int, default=2000, help="embedding dim")
parser.add_argument("--n_items", type=int, default=3589, help="number of items")
parser.add_argument("--S_n_users", type=int, default=18305, help="Source users number")
parser.add_argument("--T_n_users", type=int, default=10660, help="Target users number")
parser.add_argument("--RPE_hidden_size", type=int, default=2000, help="hidden size of Rating Pattern Extractor")
parser.add_argument("--S_pretrained_weights", type=str, default=r'pretrain\S_AutoRec.pkl')
parser.add_argument("--T_pretrained_weights", type=str, default=r'pretrain\T_AutoRec.pkl')
args = parser.parse_args()

# Load Data
dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv', preprocessed=True)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

val_df = pd.read_csv("dataset/data_target_val_processed.csv")
test_df = pd.read_csv("dataset/data_test_processed.csv")

print("Data is loaded")
# neural network
net = DArec(args)
net.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=args.wd, lr=args.lr)
RMSE = MRMSELoss().cuda()
criterion = DArec_Loss().cuda()

# loss, source_mask, target_mask
def train(epoch):
    Total_RMSE = 0
    Total_MASK = 0
    print("Epoch", epoch+1)
    predictions = []
    for idx, d in enumerate(tqdm(dataloader)):
        # alpha referring DDAN
        p = float(idx + epoch * len(dataloader)) / args.batch_size / len(dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # data
        source_rating, target_rating, source_labels, target_labels = d

        source_rating = source_rating.cuda()
        target_rating = target_rating.cuda()
        source_labels = source_labels.squeeze(1).long().cuda()
        target_labels = target_labels.squeeze(1).long().cuda()

        optimizer.zero_grad()
        is_source = True
        if is_source:
            class_output, source_prediction, target_prediction = net(source_rating, alpha, is_source)
            source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                              source_rating, target_rating, source_labels)

            rmse, _ = RMSE(target_prediction, target_rating)

            Total_RMSE += rmse.item()
            Total_MASK += torch.sum(target_mask).item()

            loss = source_loss
            predictions.extend(target_prediction.cpu().detach().numpy())
        is_source = False
        if not is_source:
            class_output, source_prediction, target_prediction = net(target_rating, alpha, is_source)
            target_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                              source_rating, target_rating, target_labels)
            loss += target_loss

        loss.backward()
        optimizer.step()

    return math.sqrt(Total_RMSE / Total_MASK), predictions


def val(predictions):
    hits1 = 0
    hits5 = 0
    hits10 = 0
    hits15 = 0
    ndcg1 = 0
    ndcg5 = 0
    ndcg10 = 0
    ndcg15 = 0
    user_interactions = downsampling(val_df, dataset.T_data)
    n_users = len(user_interactions)
    for user, user_items in tqdm(user_interactions):
        user_pred = [float(pred[user]) for pred in predictions]
        item_scores = np.array(user_pred)[user_items]
        item_ranking = np.argsort(item_scores)
        item_ranking = item_ranking[::-1]
        # The target item is the first item
        target_ranking = item_ranking.tolist().index(0) + 1
        if target_ranking <= 15:
            hits15 += 1
            ndcg15 += (1 / np.log2(target_ranking+1))
        if target_ranking <= 10:
            hits10 += 1
            ndcg10 += (1 / np.log2(target_ranking+1))
        if target_ranking <= 5:
            hits5 += 1
            ndcg5 += (1 / np.log2(target_ranking+1))
        if target_ranking <= 1:
            hits1 += 1
            ndcg1 += (1 / np.log2(target_ranking+1))

    return (hits1/n_users, hits5/n_users, hits10/n_users, hits15/n_users,
            ndcg1/n_users, ndcg5/n_users, ndcg10/n_users, ndcg15/n_users)


def test(predictions):
    net.load_state_dict(torch.load(r"weights/best_model.pkl"))
    hits1 = 0
    hits5 = 0
    hits10 = 0
    hits15 = 0
    ndcg1 = 0
    ndcg5 = 0
    ndcg10 = 0
    ndcg15 = 0
    user_interactions = downsampling(test_df, dataset.T_data)
    n_users = len(user_interactions)
    for user, user_items in tqdm(user_interactions):
        user_pred = [float(pred[user]) for pred in predictions]
        item_scores = np.array(user_pred)[user_items]
        item_ranking = np.argsort(item_scores)
        item_ranking = item_ranking[::-1]
        # The target item is the first item
        target_ranking = item_ranking.tolist().index(0) + 1
        if target_ranking <= 15:
            hits15 += 1
            ndcg15 += (1 / np.log2(target_ranking + 1))
        if target_ranking <= 10:
            hits10 += 1
            ndcg10 += (1 / np.log2(target_ranking + 1))
        if target_ranking <= 5:
            hits5 += 1
            ndcg5 += (1 / np.log2(target_ranking + 1))
        if target_ranking <= 1:
            hits1 += 1
            ndcg1 += (1 / np.log2(target_ranking + 1))

    return (hits1 / n_users, hits5 / n_users, hits10 / n_users, hits15 / n_users,
            ndcg1 / n_users, ndcg5 / n_users, ndcg10 / n_users, ndcg15 / n_users)


if __name__ == "__main__":
    train_rmses = []
    val_hits1 = []
    val_hits5 = []
    val_hits10 = []
    val_hits15 = []
    val_ndcg1 = []
    val_ndcg5 = []
    val_ndcg10 = []
    val_ndcg15 = []
    wdir = r"weights\\"
    best_metric = 0
    predictions = []
    for epoch in range(args.epochs):
        train_loss, predictions = train(epoch)
        train_rmses.append(train_loss)
        hits1, hits5, hits10, hits15, ndcg1, ndcg5, ndcg10, ndcg15 = val(predictions)
        val_hits1.append(hits1)
        val_hits5.append(hits5)
        val_hits10.append(hits10)
        val_hits15.append(hits15)
        val_ndcg1.append(ndcg1)
        val_ndcg5.append(ndcg5)
        val_ndcg10.append(ndcg10)
        val_ndcg15.append(ndcg15)
        if hits15 > best_metric:
            best_metric = hits15
            torch.save(net.state_dict(), wdir + "best_model.pkl")
        print("==================================")
        print("Validation hits1:", hits1)
        print("Validation hits5", hits5)
        print("Validation hits10", hits10)
        print("Validation hits15", hits15)
        print("Validation ndcg1", ndcg1)
        print("Validation ndcg5", ndcg5)
        print("Validation ndcg10", ndcg10)
        print("Validation ndcg15", ndcg15)

    hits1, hits5, hits10, hits15, ndcg1, ndcg5, ndcg10, ndcg15 = test(predictions)

    np.save(r"log\train_rmse", np.array(train_rmses))
    np.save(r"log\val_hits1", np.array(val_hits1))
    np.save(r"log\val_hits5", np.array(val_hits5))
    np.save(r"log\val_hits10", np.array(val_hits10))
    np.save(r"log\val_hits15", np.array(val_hits15))
    np.save(r"log\val_ndcg1", np.array(val_ndcg1))
    np.save(r"log\val_ndcg5", np.array(val_ndcg5))
    np.save(r"log\val_ndcg10", np.array(val_ndcg10))
    np.save(r"log\val_ndcg15", np.array(val_ndcg15))

    test_result = {
        "test_hits1": hits1,
        "test_hits5": hits5,
        "test_hits10": hits10,
        "test_hits15": hits15,
        "test_ndcg1": ndcg1,
        "test_ndcg5": ndcg5,
        "test_ndcg10": ndcg10,
        "test_ndcg15": ndcg15,
    }

    with open(r"log/test_result.json", "w") as file:
        json.dump(test_result, file)

    print(test_result)

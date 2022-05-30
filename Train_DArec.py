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
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='DArec with PyTorch')
parser.add_argument('--epochs', '-e', type=int, default=20)
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-5)
parser.add_argument("--n_factors", type=int, default=200, help="embedding dim")
parser.add_argument("--n_items", type=int, default=17241, help="size of each batch")
parser.add_argument("--S_n_users", type=int, default=24260, help="Source items number")
parser.add_argument("--T_n_users", type=int, default=72957, help="Target items number")
parser.add_argument("--RPE_hidden_size", type=int, default=200, help="hidden size of Rating Pattern Extractor")
parser.add_argument("--S_pretrained_weights", type=str, default=r'pretrain\S_AutoRec.pkl')
parser.add_argument("--T_pretrained_weights", type=str, default=r'pretrain\T_AutoRec.pkl')
args = parser.parse_args()

# Load Data
train_dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv',
                       train=True, preprocessed=True)
val_dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv',
                     train=False, preprocessed=True)
total_dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv',
                       train_ratio=1, val_ratio=0, train=True, preprocessed=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
total_loader = DataLoader(total_dataset, batch_size=1, shuffle=False)

val_df = pd.read_csv("dataset/data_target.csv.processed.csv")
val_df = val_df.iloc[val_dataset.val_indices, :]
val_df.reset_index(drop=True, inplace=True)
test_df = pd.read_csv("dataset/data_test.csv.processed.csv")


print("Data is loaded")
# neural network
net = DArec(args)
net.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
net.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
net.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=args.wd, lr=args.lr)
RMSE = MRMSELoss().cuda()
criterion = DArec_Loss().cuda()


# loss, source_mask, target_mask
def train(epoch):
    # process = []
    Total_RMSE = 0
    Total_MASK = 0
    print("Epoch", epoch+1)
    for idx, d in enumerate(tqdm(train_loader)):
        # alpha referring DDAN
        p = float(idx + epoch * len(train_loader)) / args.batch_size / len(train_loader)
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
        is_source = False
        if not is_source:
            class_output, source_prediction, target_prediction = net(target_rating, alpha, is_source)
            target_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                              source_rating, target_rating, target_labels)
            loss += target_loss

        loss.backward()
        optimizer.step()

    return math.sqrt(Total_RMSE / Total_MASK)


def val():
    Total_RMSE = 0
    Total_MASK = 0
    with torch.no_grad():
        for idx, d in enumerate(tqdm(val_loader)):
            # alpha referring DDAN
            p = float(idx + epoch * len(train_loader)) / args.batch_size / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # data
            source_rating, target_rating, source_labels, target_labels = d

            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()

            is_source = True
            if is_source:
                class_output, source_prediction, target_prediction = net(source_rating, alpha, is_source)
                source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                                  source_rating, target_rating, source_labels)
                rmse, _ = RMSE(target_prediction, target_rating)
                Total_RMSE += rmse.item()
                Total_MASK += torch.sum(target_mask).item()
        predictions = []
        for idx, d in enumerate(tqdm(total_loader)):
            # data
            source_rating, target_rating, source_labels, target_labels = d

            source_rating = source_rating.cuda()

            is_source = True
            if is_source:
                class_output, source_prediction, target_prediction = net(source_rating, alpha, is_source)
                target_prediction = target_prediction.squeeze(0).cpu().numpy()
                target_prediction[target_prediction <= 0.5] = 0
                target_prediction[target_prediction > 0.5] = 1
                predictions.append(target_prediction)
        pred_interactions = []
        labels = []
        for index, row in enumerate(val_df.iterrows()):
            row = row[1]
            user_idx, item_idx, label = row["user"], row["item"], row["label"]
            labels.append(label)
            pred_interactions.append(predictions[item_idx][user_idx])
        val_acc = accuracy_score(np.array(pred_interactions), np.array(labels))

    return math.sqrt(Total_RMSE / Total_MASK), val_acc


def test():
    net.load_state_dict(torch.load(r"weights/best_model.pkl"))
    Total_RMSE = 0
    Total_MASK = 0
    with torch.no_grad():
        predictions = []
        for idx, d in enumerate(tqdm(total_loader)):
            # alpha referring DDAN
            p = float(idx + epoch * len(train_loader)) / args.batch_size / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # data
            source_rating, target_rating, source_labels, target_labels = d

            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()

            is_source = True
            if is_source:
                class_output, source_prediction, target_prediction = net(source_rating, alpha, is_source)
                source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                                  source_rating, target_rating, source_labels)
                rmse, _ = RMSE(target_prediction, target_rating)
                Total_RMSE += rmse.item()
                Total_MASK += torch.sum(target_mask).item()
                target_prediction = target_prediction.squeeze(0).cpu().numpy()
                target_prediction[target_prediction <= 0.5] = 0
                target_prediction[target_prediction > 0.5] = 1
                predictions.append(target_prediction)
        pred_interactions = []
        labels = []
        for index, row in enumerate(test_df.iterrows()):
            row = row[1]
            user_idx, item_idx, label = row["user"], row["item"], row["label"]
            labels.append(label)
            pred_interactions.append(predictions[item_idx][user_idx])
        test_acc = accuracy_score(np.array(pred_interactions), np.array(labels))

    return math.sqrt(Total_RMSE / Total_MASK), test_acc


if __name__ == "__main__":
    train_rmses = []
    val_rmses = []
    val_accs = []
    wdir = r"weights\\"
    best_acc = 0
    for epoch in range(args.epochs):
        train_rmses.append(train(epoch))
        val_rmse, val_acc = val()
        val_rmses.append(val_rmse)
        val_accs.append(val_acc)
        if val_acc > best_acc:
            torch.save(net.state_dict(), wdir + "best_model.pkl")
        print("Validation loss:", val_rmse)
        print("Validation Accuracy", val_acc)

    test_loss, test_acc = test()

    np.save(r"log\train_rmse", np.array(train_rmses))
    np.save(r"log\val_rmse", np.array(val_rmses))
    np.save(r"log\val_accs", np.array(val_accs))

    test_result = {
        "test_loss": test_loss,
        "test_acc": test_acc
    }

    with open(r"log/test_result.json", "w") as file:
        json.dump(test_result, file)

    print(train_rmses)
    print(val_rmses)
    print(val_accs)
    print(test_loss, test_acc)

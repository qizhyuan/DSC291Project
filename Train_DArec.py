import numpy as np
import torch.optim as optim
import torch.utils.data
from model import *
from torch.utils.data import DataLoader
from Data_Preprocessing import Mydata
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

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
                       train=True, val=False, test=False, preprocessed=True)
val_dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv',
                     train=False, val=True, test=False, preprocessed=True)
test_dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv',
                      train=False, val=False, test=True, preprocessed=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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
    for idx, d in enumerate(train_loader):
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
        for idx, d in enumerate(val_loader):
            # alpha referring DDAN
            p = float(idx + epoch * len(train_loader)) / args.batch_size / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # data
            source_rating, target_rating, source_labels, target_labels = d

            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()
            target_labels = target_labels.squeeze(1).long().cuda()

            is_source = True
            if is_source:
                class_output, source_prediction, target_prediction = net(source_rating, alpha, is_source)
                source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                                  source_rating, target_rating, source_labels)
                rmse, _ = RMSE(target_prediction, target_rating)
                # rmse, _ = RMSE(source_prediction, source_rating)
                Total_RMSE += rmse.item()
                Total_MASK += torch.sum(target_mask).item()

                loss = source_loss

    return math.sqrt(Total_RMSE / Total_MASK)


def test():
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            # alpha referring DDAN
            p = float(idx + epoch * len(test_loader)) / args.batch_size / len(test_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # data
            rating = d

            rating = rating.cuda()

            _, _, target_prediction = net(rating, alpha, True)

    return 0


if __name__ == "__main__":
    train_rmse = []
    test_rmse = []
    wdir = r"weights\\"

    for epoch in tqdm(range(args.epochs)):
        train_rmse.append(train(epoch))
        test_rmse.append(val())
        if epoch % args.epochs == args.epochs - 1:
            torch.save(net.state_dict(), wdir + "%d.pkl" % (epoch + 1))

    np.save(r"log\train_rmse", np.array(train_rmse))
    np.save(r"log\val_rmse", np.array(train_rmse))

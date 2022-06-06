from torch import nn, optim
from AutoRec import *
from dataset import Mydata
from function import MRMSELoss
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
import math


def check_positive(val):
    val = int(val)
    if val <= 0:
        raise argparse.ArgumentError(f'{val} is invalid value. epochs should be positive integer')
    return val


def str2bool(v):
    if isinstance(v,bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='AutoRec with PyTorch')
parser.add_argument('--epochs', '-e', type=check_positive, default=30)
parser.add_argument('--batch_size', '-b', type=check_positive, default=64)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-4)
parser.add_argument('--n_factors', type=int, help="embedding size of autoencoder", default=2000)
parser.add_argument('--train_S', type=str2bool, help="Whether to train the source autoencoder", default=True)
args = parser.parse_args()

train_dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv',
                       preprocessed=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

print("Data is loaded")

if args.train_S:
    n_items, n_users = train_dataset.S_data.shape[0], train_dataset.S_data.shape[1]
else:
    n_items, n_users = train_dataset.T_data.shape[0], train_dataset.T_data.shape[1]

model = I_AutoRec(n_users=n_users, n_items=n_items, n_factors=args.n_factors).cuda()
criterion = MRMSELoss().cuda()

optimizer = optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)


def train(epoch):
    model.train()
    Total_RMSE = 0
    Total_MASK = 0
    loc = 0 if args.train_S else 1
    for idx, d in enumerate(train_loader):
        data = d[loc].cuda()
        optimizer.zero_grad()
        _, pred = model(data)
        pred.cuda()
        loss, mask = criterion(pred, data)
        Total_RMSE += loss.item()
        Total_MASK += torch.sum(mask).item()
        loss.backward()
        optimizer.step()

    return math.sqrt(Total_RMSE / Total_MASK)


if __name__ == "__main__":
    train_rmse = []
    wdir = r"pretrain\\"
    model_name = r'S_AutoRec' if args.train_S else r'T_AutoRec'
    min_loss = 999999999999
    for epoch in tqdm(range(args.epochs)):
        train_loss = train(epoch)
        train_rmse.append(train_loss)
        if train_loss < min_loss:
            torch.save(model.state_dict(), wdir + model_name + ".pkl")

    np.save(r"log\autoencoder_{}_loss.npy".format("S" if args.train_S else "T"), train_rmse)

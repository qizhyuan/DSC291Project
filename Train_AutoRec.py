from torch import nn, optim
from AutoRec import *
from dataset import Mydata
from function import MRMSELoss
from torch.utils.data import DataLoader, Dataset
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


def check_positive(val):
    val = int(val)
    if val <= 0:
        raise argparse.ArgumentError(f'{val} is invalid value. epochs should be positive integer')
    return val


parser = argparse.ArgumentParser(description='AutoRec with PyTorch')
parser.add_argument('--epochs', '-e', type=check_positive, default=20)
parser.add_argument('--batch_size', '-b', type=check_positive, default=64)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-4)
parser.add_argument('--n_factors', type=int, help="embedding size of autoencoder", default=200)
parser.add_argument('--train_S', type=bool, help="Whether to train the source autoencoder", default=True)
args = parser.parse_args()

train_dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv',
                       train=True, preprocessed=True)
test_dataset = Mydata(r'dataset\data_source.csv', r'dataset\data_target.csv', r'dataset\data_test.csv',
                      train=False, preprocessed=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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
        # RMSE = torch.sqrt(loss.item() / torch.sum(mask))
        loss.backward()
        optimizer.step()

    return math.sqrt(Total_RMSE / Total_MASK)


def test():
    model.eval()
    Total_RMSE = 0
    Total_MASK = 0
    loc = 0 if args.train_S else 1
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            data = d[loc].cuda()
            _, pred = model(data)
            pred.cuda()
            loss, mask = criterion(pred, data)
            Total_RMSE += loss.item()
            Total_MASK += torch.sum(mask).item()

    return math.sqrt(Total_RMSE / Total_MASK)


if __name__ == "__main__":
    train_rmse = []
    test_rmse = []
    wdir = r"pretrain\\"
    model_name = r'S_AutoRec' if args.train_S else r'T_AutoRec'
    min_loss = 999999999999
    for epoch in tqdm(range(args.epochs)):
        train_rmse.append(train(epoch))
        test_loss = test()
        test_rmse.append(test_loss)
        if test_loss < min_loss:
            torch.save(model.state_dict(), wdir + model_name + ".pkl")

    plt.figure(figsize=(10, 7))
    plt.plot(range(args.epochs), train_rmse, label="Train")
    plt.plot(range(args.epochs), test_rmse, label="Validation")
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.xticks(range(0, args.epochs, 2))
    plt.legend(loc="upper right", fontsize=14)
    plt.title("Loss of AutoEncoder for %s" % ("Source domain" if args.train_S else "Target domain"), fontsize=14)
    plt.savefig(r"images/autoencoder_loss_{}.jpg".format("S" if args.train_S else "T"), dpi=300, bbox_inches="tight")
    plt.show()

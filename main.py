import random
from collections import defaultdict
import torch
from tqdm import tqdm

from source import SourceModel
from target import TargetModel
from utils import train_eval_split, load_data, compute_hit_ndcg


# def evaluation(model, data_eval, criterion, batch_size):
#     model.eval()
#     predict_list, label_list, loss_list = [], [], []
#     for idx, batch_samples in enumerate(tqdm(get_batch_data(data_eval, batch_size))):
#         batch_tensor = torch.tensor(batch_samples)
#         uids = batch_tensor[:, 0].to(device).long()
#         gids = batch_tensor[:, 1].to(device).long()
#         labels = batch_tensor[:, 2].to(device).float()
#
#         out = model(uids, gids).squeeze()
#         loss = criterion(out, labels)
#         loss_list.append(loss.item())
#         out = torch.sigmoid(out)
#         predicts = out.reshape(-1).cpu().detach().numpy().tolist()
#         labels = labels.reshape(-1).cpu().detach().numpy().tolist()
#         predict_list.extend(predicts)
#         label_list.extend(labels)
#     loss = sum(loss_list) / len(loss_list)
#     thresholds = [0.01 * i for i in range(101)]
#     best_accuracy, best_threshold = 0.0, 1.0
#     for threshold in thresholds:
#         correct_num = 0
#         for i in range(len(predict_list)):
#             if predict_list[i] >= threshold and label_list[i] == 1:
#                 correct_num += 1
#             elif predict_list[i] < threshold and label_list[i] == 0:
#                 correct_num += 1
#         accuracy = correct_num / len(predict_list)
#         if accuracy >= best_accuracy:
#             best_threshold = threshold
#             best_accuracy = accuracy
#     print(f"evaluation best threshold: {best_threshold}, best accuracy: {best_accuracy}, loss: {loss}")
#     return best_accuracy, best_threshold


def train_source_model(source_model, data_train_source, data_eval_source, items_per_user_s, source_item_list,
                       criterion_s, optimizer_s, device, epoch, batch_size, save_path):
    global_best_hits = 0.
    for i in range(epoch):
        epoch_data = negative_sampling(data_train_source, items_per_user_s, source_item_list)
        epoch_loss_list = []
        source_model.train()
        for idx, batch_samples in enumerate(tqdm(get_batch_data(epoch_data, batch_size))):
            batch_tensor = torch.tensor(batch_samples)
            uids = batch_tensor[:, 0].to(device).long()
            gids = batch_tensor[:, 1].to(device).long()
            labels = batch_tensor[:, 2].to(device).float()
            optimizer_s.zero_grad()
            out = source_model(uids, gids).squeeze()
            loss = criterion_s(out, labels)
            epoch_loss_list.append(loss.item())
            loss.backward()
            optimizer_s.step()
        epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        print(f"training epoch idx: {i}, epoch loss: {epoch_loss}")
        hits = [1, 5, 10, 15]
        hits_result, ndcg_result = compute_hit_ndcg(source_model, data_eval_source, hits)
        print(10 *"-" + "Hits@K" + 10 *"-")
        for hit in hits:
            print(f"Hits@{hit}: {hits_result[hit]}")
        print(10 * "-" + "NDCG@K" + 10 * "-")
        for hit in hits:
            print(f"NDCG@{hit}: {ndcg_result[hit]}")
        if hits_result[hits[0]] >= global_best_hits:
            global_best_hits = hits_result[hits[0]]
            source_model.save_source_embeddings(save_path)


def train_target_model(target_model, data_train_target, data_eval_target, items_per_user_t, target_item_list,
                       criterion_t, optimizer_t, device, epoch, batch_size, save_path):
    global_best_hits = 0.
    for i in range(epoch):
        epoch_data = negative_sampling(data_train_target, items_per_user_t, target_item_list)
        epoch_loss_list = []
        target_model.train()
        for idx, batch_samples in enumerate(tqdm(get_batch_data(epoch_data, batch_size))):
            batch_tensor = torch.tensor(batch_samples)
            uids = batch_tensor[:, 0].to(device).long()
            gids = batch_tensor[:, 1].to(device).long()
            labels = batch_tensor[:, 2].to(device).float()
            optimizer_t.zero_grad()
            out = target_model(uids, gids).squeeze()
            loss = criterion_t(out, labels)
            epoch_loss_list.append(loss.item())
            loss.backward()
            optimizer_t.step()
        epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        print(f"training epoch idx: {i}, epoch loss: {epoch_loss}")
        hits = [1, 5, 10, 15]
        hits_result, ndcg_result = compute_hit_ndcg(target_model, data_eval_target, hits)
        print(10 * "-" + "Hits@K" + 10 * "-")
        for hit in hits:
            print(f"Hits@{hit}: {hits_result[hit]}")
        print(10 * "-" + "NDCG@K" + 10 * "-")
        for hit in hits:
            print(f"NDCG@{hit}: {ndcg_result[hit]}")
        if hits_result[hits[0]] >= global_best_hits:
            global_best_hits = hits_result[hits[0]]
            torch.save(target_model.state_dict(), save_path)


def negative_sampling(data, items_per_user, item_list, neg_num=5):
    result = []
    for datum in data:
        result.append(datum)
        u, v, _ = datum
        for i in range(neg_num):
            v_neg = random.choice(item_list)
            while v_neg in items_per_user[u]:
                v_neg = random.choice(item_list)
            result.append((u, v_neg, 0))
    random.shuffle(result)
    return result

# generate random items for computing Hits/NDCG
def generate_evaluation_data(data, items_per_user, item_list, neg_num=99):
    result = []
    for datum in data:
        u, v, r = datum
        pos = [v]
        neg = []
        generated = set()
        for i in range(neg_num):
            v_neg = random.choice(item_list)
            while v_neg in items_per_user[u] or v_neg in generated or neg == v:
                v_neg = random.choice(item_list)
            generated.add(v_neg)
            neg.append(v_neg)
        result.append((u, pos, neg))
    return result


def get_batch_data(data, batch_size):
    random.shuffle(data)
    num = len(data)
    batch_num = num // batch_size
    if batch_num * batch_size < num:
        batch_num += 1
    for i in range(batch_num):
        if i == batch_num - 1:
            yield data[i * batch_size:]
        else:
            yield data[i * batch_size: (i + 1) * batch_size]


if __name__ == '__main__':
    random.seed(2022)
    # source-domain data
    source_path = "./dataset/data_source.csv"
    source_data, source_user_idx_dict, source_item_idx_dict = load_data(source_path)
    data_train_source, data_eval_source = train_eval_split(source_data, 0.98)

    user_num_source = len(source_user_idx_dict)
    item_num_source = len(source_item_idx_dict)

    source_user_list = [i for i in range(user_num_source)]
    source_item_list = [i for i in range(item_num_source)]

    # target-domain data
    target_path = "./dataset/data_target.csv"
    target_data, user_idx_dict, item_idx_dict = load_data(target_path, source_user_idx_dict, source_item_idx_dict)
    data_train_target, _ = train_eval_split(target_data, 1.0)

    # target-domain test data
    test_path = "./dataset/data_test.csv"
    data_test, _, _ = load_data(test_path, user_idx_dict, item_idx_dict)

    user_num = len(user_idx_dict)
    item_num = len(item_idx_dict)

    # construct user-items dict for negative sampling later
    items_per_user_s = defaultdict(set)
    for (uid, gid, _) in source_data:
        items_per_user_s[uid].add(gid)

    items_per_user_t = defaultdict(set)
    target_item_set = set()
    for (uid, gid, _) in target_data:
        items_per_user_t[uid].add(gid)
        target_item_set.add(gid)

    user_list = [i for i in range(user_num)]
    item_list = [i for i in range(item_num)]

    emb_dim = 32
    batch_size_s = 128
    batch_size_t = 128

    # evaluation data for source domain
    data_eval_source = generate_evaluation_data(data_eval_source, items_per_user_s, item_list, 99)
    data_eval_target = generate_evaluation_data(data_test, items_per_user_t, item_list, 99)
    # data_eval_source = negative_sampling(data_eval_source, items_per_user_s, item_list, 1)

    device = torch.device("cuda")
    # learning_rate_s = 5e-3
    learning_rate = 5e-4
    weight_decay = 1e-8
    criterion = torch.nn.BCEWithLogitsLoss()

    # train single-domain model on target domain
    print("Train Target Model Without Transfer")
    target_model_baseline = SourceModel(user_num, item_num, emb_dim).to(device)
    optimizer_base = torch.optim.Adam(target_model_baseline.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_source_model(target_model_baseline, data_train_target, data_eval_target, items_per_user_s, source_item_list, criterion,
                       optimizer_base, device, 15,
                       batch_size_s, './target_model_baseline.pkl')


    # train source-domain model
    print("Train Source Model")
    source_model = SourceModel(user_num, item_num, emb_dim).to(device)
    optimizer_s = torch.optim.Adam(source_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pretrain_path = './source_model.pkl'
    train_source_model(source_model, data_train_source, data_eval_source, items_per_user_s, source_item_list, criterion,
                       optimizer_s, device, 15,
                       batch_size_s, pretrain_path)

    # train target-domain model

    print("Train Target Model With Transfer")
    target_model = TargetModel(user_num, item_num, emb_dim)
    target_model.load_source_embeddings(pretrain_path)
    target_model = target_model.to(device)
    save_path = './target_model.pkl'
    optimizer_t = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    target_item_list = list(target_item_set)
    train_target_model(target_model, data_train_target, data_eval_target, items_per_user_t, target_item_list, criterion,
                       optimizer_t, device, 15, batch_size_t, save_path)

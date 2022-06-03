import csv
import math
import random

import torch


def read_csv(path):
    with open(path, 'rt') as f:
        c = csv.reader(f)
        header = next(c)
        for line in c:
            d = dict(zip(header, line))
            yield d['user'], d['item'], d


def load_data(path, user_idx_dict=None, item_idx_dict=None):
    data = []
    if user_idx_dict is None:
        user_idx_dict = dict()
    if item_idx_dict is None:
        item_idx_dict = dict()
    for user, recipe, d in read_csv(path):
        if user not in user_idx_dict:
            user_idx_dict[user] = len(user_idx_dict)
        if recipe not in item_idx_dict:
            item_idx_dict[recipe] = len(item_idx_dict)
        uid = user_idx_dict[user]
        gid = item_idx_dict[recipe]
        if 'rating' in d:
            label = float(d['rating'])
            if label > 0:
                label = 1.
            else:
                label = 0.
            data.append((uid, gid, label))
        else:
            data.append((uid, gid, 1.0))

    return data, user_idx_dict, item_idx_dict


def train_eval_split(data, ratio=0.8):
    instance_num = len(data)
    random.shuffle(data)
    train_num = int(instance_num * ratio)
    data_train = data[:train_num]
    data_eval = data[train_num:]
    return data_train, data_eval


def tensor_post_processor(batch_data):
    uids, gids = None, None
    for item in batch_data["uid"]:
        if uids is None:
            uids = item.reshape((-1, 1))
        else:
            uids = torch.cat((uids, item.reshape((-1, 1))), dim=1)
    uids = uids.reshape(-1)
    for item in batch_data["gid"]:
        if gids is None:
            gids = item.reshape((-1, 1))
        else:
            gids = torch.cat((gids, item.reshape((-1, 1))), dim=1)
    gids = gids.reshape(-1)
    labels = None
    for item in batch_data["label"]:
        if labels is None:
            labels = item.reshape((-1, 1))
        else:
            labels = torch.cat((labels, item.reshape((-1, 1))), dim=1)
    labels = labels.reshape(-1)
    return uids, gids, labels


def compute_hit_ndcg(model, test_data, k_list=None):
    if k_list is None:
        k_list = [1]
    model.eval()
    k_list.sort()
    device = next(model.parameters()).device
    hits_result = {k: 0 for k in k_list}
    ndcg_result = {k: 0 for k in k_list}
    test_num = len(test_data)
    # test_data:[[user_id, [positive items], [negative items]]]
    for uid, pos, neg in test_data:
        pos_num = len(pos)
        gids = pos + neg
        gid_num = len(gids)
        gid_tensor = torch.tensor(gids).to(device).long()
        uid_tensor = torch.tensor([uid for _ in range(gid_num)]).to(device).long()
        out = model(uid_tensor, gid_tensor).squeeze()
        indices = torch.argsort(out, descending=True)
        hits_counter = {k: 0 for k in k_list}
        ndcg_counter = {k: 0 for k in k_list}
        for i in range(k_list[-1]):
            idx = indices[i].item()
            if idx < pos_num:
                for k in k_list:
                    if i < k:
                        hits_counter[k] += 1
                        ndcg_counter[k] += 1 / math.log2(i + 2)
        for k in k_list:
            hits_result[k] += hits_counter[k] / pos_num
            ndcg_result[k] += ndcg_counter[k] / pos_num
    for k in hits_result:
        hits_result[k] /= test_num
        ndcg_result[k] /= test_num
    return hits_result, ndcg_result

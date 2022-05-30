import csv
import random
import multiprocessing as mp
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
        if 'label' in d:
            label = float(d['label'])
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


def negative_sampling_helper(input_q: mp.Queue, out_put_q: mp.Queue, items_per_user, item_list, neg_num):
    while not input_q.empty():
        datum = input_q.get(timeout=1)
        if datum is not None:
            out_put_q.put(datum)
            u, v, _ = datum
            for i in range(neg_num):
                v_neg = random.choice(item_list)
                tries = 0
                while v_neg in items_per_user[u] and tries < 5:
                    v_neg = random.choice(item_list)
                    tries += 1
                out_put_q.put((u, v_neg, 0))


from collections import defaultdict


def compute_hit_ratio(model, test_data, k_list=None):
    if k_list is None:
        k_list = [1]
    model.eval()
    k_list.sort()
    test_data_real = []
    items_per_user = defaultdict(set)
    hits_dict = defaultdict(list)
    for uid, gid, label in test_data:
        if label == 0:
            continue
        test_data_real.append((uid, gid))
        items_per_user[uid].add(gid)

    uids, gids = list(zip(*test_data_real))
    uid_set, gid_set = set(uids), set(gids)
    gid_candidates = list(gid_set)
    gid_num = len(gid_candidates)
    device = next(model.parameters()).device
    gid_tensor = torch.tensor(gid_candidates).to(device).long()
    for uid in uid_set:
        uid_tensor = torch.tensor([uid for _ in range(gid_num)]).to(device).long()
        out = model(uid_tensor, gid_tensor).squeeze()
        indices = torch.argsort(out, descending=True)
        hits_record = {k: 0 for k in k_list}
        for i in range(k_list[-1]):
            idx = indices[i].item()
            gid = gid_candidates[idx]
            if gid in items_per_user[uid]:
                for k in k_list:
                    if i < k:
                        hits_record[k] += 1
        for k in hits_record:
            hits_record[k] /= len(items_per_user[uid])
        for k, v in hits_record.items():
            hits_dict[k].append(v)
    hits_result = dict()
    for k, v in hits_dict.items():
        if len(v) == 0:
            hits_result[k] = 0
        else:
            hits_result[k] = sum(v) / len(v)
    return hits_result

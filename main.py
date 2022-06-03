import random
from collections import defaultdict
import torch
import json
from tqdm import tqdm

from source import SourceModel
from target import TargetModel
from utils import train_eval_split, load_data, compute_hit_ndcg


def train_source_model(source_model, data_train_source, data_eval_source, items_per_user_s, source_item_list,
                       criterion_s, optimizer_s, device, epoch, batch_size, save_path):
    global_best_hits = 0.
    result_dict = defaultdict(list)
    for i in range(epoch):
        epoch_data = negative_sampling(data_train_source, items_per_user_s, source_item_list)
        epoch_loss_list = []
        source_model.train()
        batch_data = BatchData(epoch_data, batch_size)
        for idx, batch_samples in enumerate(tqdm(batch_data)):
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
        print(f"Training Epoch: {i + 1}, Training Loss: {epoch_loss}")
        result_dict["loss"].append(epoch_loss)
        hits = [1, 5, 10, 15]
        hits_result, ndcg_result = compute_hit_ndcg(source_model, data_eval_source, hits)
        print(10 * "-" + "Hits@K" + 10 * "-")
        for hit in hits:
            print(f"Hits@{hit}: {hits_result[hit]}")
            result_dict[f"Hits@{hit}"].append(hits_result[hit])
        print(10 * "-" + "NDCG@K" + 10 * "-")
        for hit in hits:
            print(f"NDCG@{hit}: {ndcg_result[hit]}")
            result_dict[f"NDCG@{hit}"].append(ndcg_result[hit])
        if hits_result[hits[0]] >= global_best_hits:
            global_best_hits = hits_result[hits[0]]
            source_model.save_source_embeddings(save_path)
    return result_dict


def train_target_model(target_model, data_train_target, data_eval_target, items_per_user_t, target_item_list,
                       criterion_t, optimizer_t, device, epoch, batch_size, save_path):
    global_best_hits = 0.
    result_dict = defaultdict(list)
    for i in range(epoch):
        epoch_data = negative_sampling(data_train_target, items_per_user_t, target_item_list)
        epoch_loss_list = []
        target_model.train()
        batch_data = BatchData(epoch_data, batch_size)
        for idx, batch_samples in enumerate(tqdm(batch_data)):
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
        print(f"Training Epoch: {i + 1}, Training Loss: {epoch_loss}")
        result_dict["loss"].append(epoch_loss)
        hits = [1, 5, 10, 15]
        hits_result, ndcg_result = compute_hit_ndcg(target_model, data_eval_target, hits)
        print(10 * "-" + "Hits@K" + 10 * "-")
        for hit in hits:
            print(f"Hits@{hit}: {hits_result[hit]}")
            result_dict[f"Hits@{hit}"].append(hits_result[hit])
        print(10 * "-" + "NDCG@K" + 10 * "-")
        for hit in hits:
            print(f"NDCG@{hit}: {ndcg_result[hit]}")
            result_dict[f"NDCG@{hit}"].append(ndcg_result[hit])
        if hits_result[hits[0]] >= global_best_hits:
            global_best_hits = hits_result[hits[0]]
            torch.save(target_model.state_dict(), save_path)
    return result_dict


def train_mixed_model(mixed_model, data_train, data_eval_s, data_eval_t, items_per_user, item_list,
                      criterion, optimizer, device, epoch, batch_size, save_path):
    global_best_hits = 0.
    result_dict = defaultdict(list)
    for i in range(epoch):
        epoch_data = negative_sampling(data_train, items_per_user, item_list)
        epoch_loss_list = []
        mixed_model.train()
        batch_data = BatchData(epoch_data, batch_size)
        for idx, batch_samples in enumerate(tqdm(batch_data)):
            batch_tensor = torch.tensor(batch_samples)
            uids = batch_tensor[:, 0].to(device).long()
            gids = batch_tensor[:, 1].to(device).long()
            labels = batch_tensor[:, 2].to(device).float()
            optimizer.zero_grad()
            out = mixed_model(uids, gids).squeeze()
            loss = criterion(out, labels)
            epoch_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        print(f"Training Epoch: {i + 1}, Training Loss: {epoch_loss}")
        result_dict["loss"].append(epoch_loss)
        hits = [1, 5, 10, 15]
        hits_result, ndcg_result = compute_hit_ndcg(mixed_model, data_eval_s, hits)
        print("Performance on the Source Domain")
        print(10 * "-" + "Hits@K" + 10 * "-")
        for hit in hits:
            print(f"Hits@{hit}: {hits_result[hit]}")
            result_dict[f"Hits@{hit}-S"].append(hits_result[hit])
        print(10 * "-" + "NDCG@K" + 10 * "-")
        for hit in hits:
            print(f"NDCG@{hit}: {ndcg_result[hit]}")
            result_dict[f"NDCG@{hit}-S"].append(ndcg_result[hit])
        hits_result, ndcg_result = compute_hit_ndcg(mixed_model, data_eval_t, hits)
        print("Performance on the Target Domain")
        print(10 * "-" + "Hits@K" + 10 * "-")
        for hit in hits:
            print(f"Hits@{hit}: {hits_result[hit]}")
            result_dict[f"Hits@{hit}-T"].append(hits_result[hit])
        print(10 * "-" + "NDCG@K" + 10 * "-")
        for hit in hits:
            print(f"NDCG@{hit}: {ndcg_result[hit]}")
            result_dict[f"NDCG@{hit}-T"].append(ndcg_result[hit])
        if hits_result[hits[0]] >= global_best_hits:
            global_best_hits = hits_result[hits[0]]
            torch.save(mixed_model.state_dict(), save_path)
    return result_dict

# negative sampling for training
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


class BatchData:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        random.shuffle(data)
        num = len(data)
        batch_num = num // batch_size
        if batch_num * batch_size < num:
            batch_num += 1
        self.batch_num = batch_num
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.batch_num - 1:
            self.idx += 1
            return self.data[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
        elif self.idx == self.batch_size - 1:
            self.idx += 1
            return self.data[self.idx * self.batch_size:]
        else:
            raise StopIteration

    def __len__(self):
        return self.batch_num


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

    device = torch.device("cuda")
    learning_rate = 5e-4
    weight_decay = 1e-8
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = 1

    result_dict = dict()

    # train target-domain MF model
    print("Train Target-Domain MF Model Without Transfer")
    target_model = SourceModel(user_num, item_num, emb_dim).to(device)
    optimizer_base = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pretrain_path_t = './mf_model_target.pkl'
    result = train_source_model(target_model, data_train_target, data_eval_target, items_per_user_t,
                                source_item_list,
                                criterion,
                                optimizer_base, device, epochs,
                                batch_size_s, pretrain_path_t)

    result_dict["MF_Model_Target"] = result

    # train source-domain MF model
    print("Train Source-Domain MF Model Without Transfer")
    source_model = SourceModel(user_num, item_num, emb_dim).to(device)
    optimizer_s = torch.optim.Adam(source_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pretrain_path_s = './mf_model_source.pkl'
    result = train_source_model(source_model, data_train_source, data_eval_source, items_per_user_s, source_item_list,
                                criterion,
                                optimizer_s, device, epochs,
                                batch_size_s, pretrain_path_s)
    result_dict["MF_Model_Source"] = result

    # train EMCDR model from source to target
    print("Train EMCDR Model With Transfer (S -> T)")
    target_model = TargetModel(user_num, item_num, emb_dim)
    target_model.load_source_embeddings(pretrain_path_s)
    target_model = target_model.to(device)
    save_path = './emcdr_s2t.pkl'
    optimizer_t = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    target_item_list = list(target_item_set)
    result = train_target_model(target_model, data_train_target, data_eval_target, items_per_user_t, target_item_list,
                                criterion,
                                optimizer_t, device, epochs, batch_size_t, save_path)
    result_dict["EMCDR_Model_S2T"] = result

    # train EMCDR model from target to source (swap the domain)
    print("Train EMCDR Model With Transfer (T -> S)")
    source_model = TargetModel(user_num, item_num, emb_dim)
    source_model.load_source_embeddings(pretrain_path_t)
    source_model = source_model.to(device)
    save_path = './emcdr_s2t.pkl'
    optimizer_s = torch.optim.Adam(source_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    source_item_list = list(source_item_list)
    result = train_target_model(source_model, data_train_source, data_eval_source, items_per_user_s, source_item_list,
                                criterion,
                                optimizer_s, device, epochs, batch_size_s, save_path)
    result_dict["EMCDR_Model_T2S"] = result

    # train MF model on both domain
    print("Train MF Model on Mixed Data")
    data_train_combined = data_train_source + data_train_target
    random.shuffle(data_train_combined)
    items_per_user = items_per_user_s.copy()
    items_per_user.update(items_per_user_t)
    mf_model_baseline = SourceModel(user_num, item_num, emb_dim).to(device)
    optimizer_base = torch.optim.Adam(mf_model_baseline.parameters(), lr=learning_rate, weight_decay=weight_decay)
    result = train_mixed_model(mf_model_baseline, data_train_combined, data_eval_source, data_eval_target,
                               items_per_user,
                               item_list,
                               criterion,
                               optimizer_base, device, epochs,
                               batch_size_s, './target_model_baseline_mixed.pkl')

    result_dict["MF_Model_Mixed"] = result
    with open("./result.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)

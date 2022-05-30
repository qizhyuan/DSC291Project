import torch


class TargetModel(torch.nn.Module):
    def __init__(self, user_num, item_num, emb_dim):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.emb_dim = emb_dim
        self.user_emb = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((user_num + 1, emb_dim))))
        self.item_emb = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((item_num + 1, emb_dim))))

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )

    def forward(self, uids, gids):
        user_emb = torch.index_select(self.user_emb, dim=0, index=uids)
        item_emb = torch.index_select(self.item_emb, dim=0, index=gids)
        source_item_emb = item_emb
        item_emb_detached = source_item_emb.detach()
        item_emb = self.layer(item_emb_detached)
        out = torch.sum(torch.multiply(user_emb, item_emb), dim=1)
        return out

    def load_source_embeddings(self, path):
        source_state_dict = torch.load(path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in source_state_dict.items() if k in model_dict and "item" in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

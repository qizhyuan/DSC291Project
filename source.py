import torch


class SourceModel(torch.nn.Module):
    def __init__(self, user_num, item_num, emb_dim):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.user_emb = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((user_num + 1, emb_dim))))
        self.item_emb = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((item_num + 1, emb_dim))))
        # self.user_emb = torch.nn.Embedding(user_num + 1, emb_dim)
        # self.item_emb = torch.nn.Embedding(item_num + 1, emb_dim)

    def forward(self, uids, gids):
        # user_emb = self.user_emb(uids)
        # item_emb = self.item_emb(gids)
        user_emb = torch.index_select(self.user_emb, dim=0, index=uids)
        item_emb = torch.index_select(self.item_emb, dim=0, index=gids)
        out = torch.sum(torch.multiply(user_emb, item_emb), dim=1)
        return out

    def save_source_embeddings(self, path):
        model_state_dict = self.state_dict()
        torch.save(model_state_dict, path)

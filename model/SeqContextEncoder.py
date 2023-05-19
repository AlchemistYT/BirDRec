import torch
from torch import nn
# from model_utils.Utils import normalize

from model.BirDRec import ContextEncoder


class SeqContextEncoder(ContextEncoder):

    def __init__(self, config, graph):
        super().__init__(config, graph)
        self.seq_module = None
        self.fc_embed = nn.Linear(config['hidden_size'] * 2, config['hidden_size'], bias=False).double()
        self.network_param_init(config)

    def network_param_init(self, config):
        pass

    def seq_modelling(self, hist_item_ids, user_embed=None, rectify=True):
        pass

    def forward(self, user_id, hist_item_ids, rectify=True):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        """
        # [bs, seq_len, hidden_size]
        # user_embed = self.user_embeddings[user_id]
        # [bs, hidden_size]
        if rectify:
            context_embed_last, context_embed_second_last = self.seq_modelling(hist_item_ids, rectify=rectify)
            user_embed = self.user_embeddings.weight[user_id]
            context_embed_last = context_embed_last + user_embed
            context_embed_second_last = context_embed_second_last + user_embed
            return context_embed_last, context_embed_second_last
        else:
            context_embed = self.seq_modelling(hist_item_ids, rectify=False)
            user_embed = self.user_embeddings.weight[user_id]
            context_embed = context_embed + user_embed
            return context_embed




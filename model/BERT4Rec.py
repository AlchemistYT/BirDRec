from model.SeqContextEncoder import SeqContextEncoder
from model_utils.BERT_SeqRec import BertModel
from model_utils.BERT_SeqRec import BertConfig
import torch


class BERT4Rec(SeqContextEncoder):

    def __init__(self, config, graph):
        super().__init__(config, graph)

    def network_param_init(self, config):
        bert_config = BertConfig(config['item_num'], config)
        self.seq_module = BertModel(bert_config, use_outer_embed=True)

    def seq_modelling(self, hist_item_ids, user_embed=None, rectify=True):
        # [bs, seq_len, hidden_size]
        bert_context = self.seq_module(hist_item_ids, outer_embed=self.item_embeddings.weight, unidirectional=False)
        # [bs, hidden_size]
        if rectify:
            return bert_context[:, -1, :].squeeze(1), bert_context[:, -2, :].squeeze(1)
        else:
            return bert_context[:, -1, :].squeeze(1)
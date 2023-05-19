from model.SeqContextEncoder import SeqContextEncoder
from torch import nn
import torch
from torch.autograd import Variable


class BERD_HGN(SeqContextEncoder):

    def __init__(self, config):
        super().__init__(config)

    def network_param_init(self, config):
        dims = config['hidden_size']
        device = config['device']
        self.feature_gate_item = nn.Linear(dims, dims).to(device)
        self.feature_gate_user = nn.Linear(dims, dims).to(device)

        self.instance_gate_item = Variable(torch.zeros(dims, 1).type(torch.DoubleTensor), requires_grad=True).to(device)
        self.instance_gate_user = Variable(torch.zeros(dims, config['input_len']).type(torch.DoubleTensor), requires_grad=True).to(device)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)

        self.W2 = nn.Embedding(config['item_num'], config['hidden_size'], padding_idx=0).to(device)
        self.b2 = nn.Embedding(config['item_num'], 1, padding_idx=0).to(device)


    def seq_modelling(self, hist_item_ids, user_embed=None):
        # user_embed = [bs, hidden_size]

        # [bs, seq_len, hidden_size]
        hist_item_embeds = self.item_embeddings.weight[hist_item_ids]

        # feature gating
        gate = torch.sigmoid(self.feature_gate_item(hist_item_embeds) + self.feature_gate_user(user_embed).unsqueeze(1))
        # [bs, seq_len, hidden_size]
        gated_item = hist_item_embeds * gate

        # instance gating
        # [bs, seq_len]
        instance_score = torch.sigmoid(torch.matmul(gated_item, self.instance_gate_item.unsqueeze(0)).squeeze() +
                                       user_embed.mm(self.instance_gate_user))
        # [bs, seq_len, hidden_size]
        union_out = gated_item * instance_score.unsqueeze(2)
        # [bs, hidden_size]
        union_out = torch.sum(union_out, dim=1)
        union_out = union_out / torch.sum(instance_score, dim=1).unsqueeze(1)

        return union_out + user_embed + torch.sum(hist_item_embeds, dim=1)


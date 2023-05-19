from model.SeqContextEncoder import SeqContextEncoder
from torch import nn
import torch
import torch.nn.functional as F


class BERD_Caser(SeqContextEncoder):

    def __init__(self, config, graph):
        super().__init__(config, graph)

    def network_param_init(self, config):
        self.n_h = config['n_h']
        self.n_v = config['n_v']
        self.drop_ratio = config['drop_ratio']
        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (config['input_len'], 1))
        # horizontal conv layer
        lengths = [i + 1 for i in range(config['input_len'])]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, config['hidden_size'])) for i in lengths]).double()
        # fully-connected layer
        self.fc1_dim_v = self.n_v * config['hidden_size']
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, config['hidden_size']).double()
        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)


    def cnn_process(self, item_image):
        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_image)
            # [bs, n_v * hidden_size]
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect
        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = F.relu(conv(item_image).squeeze(3))
                # [bs, n_h]
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            # [bs, n_h * input_len]
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        return self.fc1(torch.cat([out_v, out_h], 1))

    def seq_modelling(self, hist_item_ids, user_embed=None, rectify=True):
        hist_item_embed = self.item_embeddings.weight[hist_item_ids]
        # Embedding Look-up
        item_image_last = hist_item_embed.unsqueeze(1)  # [bs, 1, seq_len, hidden_size]
        context_embed_last = self.cnn_process(item_image_last)

        if rectify:
            second_last_hist_item_ids = torch.concat(
                [torch.zeros(size=[hist_item_ids.shape[0], 1]).to(self.device).long(), hist_item_ids[:, 1:]], dim=1)
            second_last_hist_item_embed = self.item_embeddings.weight[second_last_hist_item_ids]
            item_image_second_last = second_last_hist_item_embed.unsqueeze(1)
            context_embed_second_last = self.cnn_process(item_image_second_last)
            return context_embed_last, context_embed_second_last
        else:
            return context_embed_last


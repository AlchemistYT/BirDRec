# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
import sys

if sys.version_info[0] >= 3:
    unicode = str

from io import open

import torch
from torch import nn

logger = logging.getLogger(__name__)

BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size, config):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_act = config['hidden_act']
        self.intermediate_size = config['intermediate_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.max_position_embeddings = config['max_seq_len']
        self.type_vocab_size = config['type_vocab_size']
        self.initializer_range = config['initializer_range']
        self.use_item_bias = True


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layerNorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            # x: [batch_size, seq_len, hidden_size]
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.

       raw_ids [batch_size, seq_len]
       -> (Position embedding + LayerNorm + Dropout)
       -> embeddings [[batch_size, seq_len, hidden_size]]
    """

    def __init__(self, config, use_outer_embed=False):
        super(BertEmbeddings, self).__init__()
        self.use_outer_embed = use_outer_embed
        if use_outer_embed is False:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
            self.outer_emb = False
        else:
            self.word_embeddings = None
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, outer_emb=None):
        # input_ids = [batch_size, seq_len]
        # token_type_ids = [batch_size, seq_len], all 0 for SeqRec
        seq_length = input_ids.size(1)
        # [seq_len]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # [batch_size, seq_len]
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # [batch_size, seq_len, hidden_size]
        if outer_emb is not None:
            words_embeddings = outer_emb[input_ids]
        else:
            words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        assert self.all_head_size == config.hidden_size

        # [hidden_size, hidden_size]
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # x:[batch_size, seq_len, hidden_size]
        # new shape: [batch_size, seq_len, num_head, head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # [batch_size, num_head, seq_len, head_size]
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # hidden_states = [batch_size, seq_len, hidden_size]
        # attention_mask = [batch_size, 1, 1, to_seq_length]

        # mixed_query_layer = [batch_size, seq_len, hidden_size]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # [batch_size, num_head, seq_len, head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # [batch_size, num_head, seq_len, head_size] * [batch_size, num_head, head_size, seq_len]
        # = [batch_size, num_head, seq_len, seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [batch_size, num_head, seq_len, seq_len]
        attention_probs = self.dropout(attention_probs)
        # [batch_size, num_head, seq_len, seq_len] * [batch_size, num_head, seq_len, head_size]
        # = [batch_size, num_head, seq_len, head_size]
        context_layer = torch.matmul(attention_probs, value_layer)
        # = [batch_size, seq_len, num_head, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # [batch_size, seq_len, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # input_tensor : [batch_size, seq_len, hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # residual layer
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self_atten = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        # hidden_states = [batch_size, seq_len, hidden_size]
        # attention_mask = [batch_size, 1, 1, to_seq_length]

        # self_output: [batch_size, seq_len, hidden_size]
        self_output = self.self_atten(input_tensor, attention_mask)
        # [batch_size, seq_len, hidden_size] (transform and residual)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        # [batch_size, seq_len, hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # [batch_size, seq_len, hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    Multi-Head self attention + Intermediate transform + output
    """

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        # hidden_states = [batch_size, seq_len, hidden_size]
        # attention_mask = [batch_size, 1, 1, to_seq_length]

        # [batch_size, seq_len, hidden_size]
        attention_output = self.attention(hidden_states, attention_mask)
        # non-linear activation -> [batch_size, seq_len, hidden_size]
        intermediate_output = self.intermediate(attention_output)
        # transform -> dropout -> LayerNorm [batch_size, seq_len, hidden_size]
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """
    A stack of BERT Layers
    """

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # hidden_states = [batch_size, seq_len, hidden_size]
        # attention_mask = [batch_size, 1, 1, to_seq_length]
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    """

    def __init__(self, config, use_outer_embed=False):
        super(BertModel, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config, use_outer_embed=use_outer_embed)
        self.outer_embed = use_outer_embed
        self.predictor = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, outer_embed=None, unidirectional=True):
        # input_ids = [batch_size, sequence_length]
        # attention_mask = [batch_size, to_seq_length]
        attention_mask = (input_ids > 0).long()

        # Create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # -------------

        # -------------

        # mask for left-to-right unidirectional -----
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)

        if unidirectional: # mask for left-to-right unidirectional -----
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(input_ids.device)
            extended_attention_mask = extended_attention_mask * subsequent_mask
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Position embedding + LayerNorm + Dropout -> [batch_size, seq_len, hidden_size]
        embedding_output = self.embeddings(input_ids, token_type_ids, outer_embed)
        encoded_layers = self.predictor(embedding_output,
                                        extended_attention_mask,
                                        output_all_encoded_layers=output_all_encoded_layers)
        # sequence_output = [batch_size, seq_len, hidden_size]
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPredictor(nn.Module):
    """
    Retrieve the predicted context embeddings corresponding to the masked positions

    bert_context: [bs, seq_len, hidden_size]
    masked_positions: [bs, max_pred_num]
    masked_pos_ids: [bs, max_pred_num]
    label_weights: [bs, max_pred_num]

    output: pred_probs [bs * max_pred_num, word_num]
    """

    def __init__(self, config):
        super(BertPredictor, self).__init__()
        # transform
        self.transform_weight = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, bert_context, masked_positions):
        # Get shapes
        batch_size = bert_context.size()[0]
        seq_length = bert_context.size()[1]
        hidden_size = bert_context.size()[2]

        # Collect target context
        # flat_context: [bs * seq_len, hidden_size]
        flat_context = bert_context.view([batch_size * seq_length, hidden_size])
        # [bs, 1]
        offsets = (torch.arange(0, batch_size) * seq_length).view([batch_size, 1])
        # [bs * max_pred_num] <- [bs, max_pred_num] + [bs, 1]
        flat_positions = (masked_positions + offsets).view([-1])
        # target_context: [bs * max_pred_num, hidden_size]
        target_context = torch.index_select(flat_context, dim=0, index=flat_positions)

        # Transform
        # transformed_context: [bs * max_pred_num, hidden_size]
        transformed_context = self.transform_weight(target_context)

        return transformed_context


class LossForBERT(nn.Module):
    """
    pred_context, [bs * max_pred_num, hidden_size]
    label_ids, [bs * max_pred_num]
    negative_ids_list, [neg_num, bs * max_pred_num]
    label_weights, [bs * max_pred_num, 1]
    word_weights, [word_num, hidden_size]
    """

    def __init__(self, config, word_weights):
        super(LossForBERT, self).__init__()
        self.loss_type = config.loss_type
        self.item_bias = None
        if config.use_item_bias:
            self.item_bias = nn.Parameter(torch.zeros(word_weights.size(0)))
        self.word_weights = word_weights
        # if config.loss_type is 'full_cross_entropy':
        #     self.word_weights = word_weights.transpose(0, 1)
        # else:
        #     self.word_weights = word_weights

    def forward(self, pred_context, label_ids, negative_ids_list, label_weights):
        loss = 0
        neg_num = len(negative_ids_list)

        pos_embed = torch.index_select(self.word_weights, dim=0, index=label_ids)
        # [bs * max_pred_num]
        pos_score = torch.mul(pred_context, pos_embed).sum(dim=1, keepdim=False)
        if self.item_bias is not None:
            # [bs * max_pred_num]
            pos_bias = torch.index_select(self.item_bias, dim=0, index=label_ids)
            pos_score += pos_bias

        for i in range(neg_num):
            neg_embed = torch.index_select(self.word_weights, dim=0, index=negative_ids_list[i])
            # [bs * max_pred_num]
            neg_score = torch.mul(pred_context, neg_embed).sum(dim=1, keepdim=False)
            if self.item_bias is not None:
                # [bs * max_pred_num]
                neg_bias = torch.index_select(self.item_bias, dim=0, index=negative_ids_list[i])
                neg_score += neg_bias
            # print(f'pos_score: {pos_score.size()}')
            # print(f'pos_score: {neg_score.size()}')
            # print(f'pos_score: {label_weights.size()}')
            loss += -torch.log(torch.sigmoid(pos_score - neg_score)).dot(label_weights.float())
        loss = loss / neg_num
        loss = loss / torch.sum(label_weights)
        return loss

    def get_aug_loss(self, pred_context, pos_augs, neg_augs, aug_weights):
        """
        pred_context, [bs * max_pred_num, hidden_size]
        pos_augs: [bs * max_pred_num, word_num], one-hot
        neg_augs: [bs * max_pred_num, word_num], one-hot
        aug_weights: [bs * max_pred_num], prob dist
        """
        # if self.loss_type is not 'full_cross_entropy':
        #     # [word_num, hidden_size]
        #     word_weights = self.word_weights
        # else:
        #     word_weights = self.word_weights.transpose(0, 1)
        # get embeddings: [bs * max_pred_num, hidden_size]
        word_weights = self.word_weights
        pos_embed = torch.mm(pos_augs, word_weights)
        neg_embed = torch.mm(neg_augs, word_weights)

        # [bs * max_pred_num]
        pos_score = torch.mul(pred_context, pos_embed).sum(dim=1, keepdim=False)
        neg_score = torch.mul(pred_context, neg_embed).sum(dim=1, keepdim=False)

        if self.item_bias is not None:
            item_bias = self.item_bias.unsqueeze(0)
            # [bs * max_pred_num, word_num] * [1, word_num]->[bs * max_pred_num]
            pos_bias = torch.mul(pos_augs, item_bias).sum(dim=1, keepdim=False)
            neg_bias = torch.mul(neg_augs, item_bias).sum(dim=1, keepdim=False)
            pos_score += pos_bias
            neg_score += neg_bias
        # [bs * max_pred_num] dot [bs * max_pred_num] -> [1]
        loss = -torch.log(torch.sigmoid(pos_score - neg_score)).dot(aug_weights)
        loss = loss / torch.sum(aug_weights)

        return loss


    def predict(self, context_embeds, target_ids):
        """
            pred_context, [bs * max_pred_num, hidden_size]
            target_ids, [bs * max_pred_num]
        """
        # [bs * max_pred_num, hidden_size]
        target_embed = torch.index_select(self.word_weights, dim=0, index=target_ids)
        # [bs * max_pred_num]
        pred_score = torch.mul(context_embeds, target_embed).sum(dim=1, keepdim=False)
        if self.item_bias is not None:
            # [bs * max_pred_num]
            target_bias = torch.index_select(self.item_bias, dim=0, index=target_ids)
            pred_score += target_bias
        return pred_score

    def pairwise_sample(self, pred_context, label_ids, negative_ids_list, label_weights):
        loss = 0
        neg_num = len(negative_ids_list)
        # [bs * max_pred_num]
        pos_score = self.predict(pred_context, label_ids)
        for i in range(neg_num):
            neg_score = self.predict(pred_context, negative_ids_list[i])
            loss += -torch.log(torch.sigmoid(pos_score - neg_score)).dot(label_weights.float())
        loss = loss / neg_num
        loss = loss / torch.sum(label_weights)
        return loss

    def pointwise_sample(self, pred_context, label_ids, negative_ids_list, label_weights):
        pos_score = self.predict(pred_context, label_ids)
        pos_loss = -torch.log(torch.sigmoid(pos_score)).dot(label_weights)

        neg_num = len(negative_ids_list)
        neg_loss = 0
        for i in range(neg_num):
            neg_score = self.predict(pred_context, negative_ids_list[i])
            neg_loss += -torch.log(1 - torch.sigmoid(neg_score)).dot(label_weights)

        loss = (pos_loss + neg_loss) / torch.sum(label_weights)
        return loss

    def full_cross_entropy(self, pred_context, label_ids, label_weights):
        # Predict
        # [bs * max_pred_num, hidden_size] * [hidden_size, word_num] -> [bs * max_pred_num, word_num]
        pred_scores = torch.mm(pred_context, self.word_weights)
        if self.item_bias is not None:
            pred_scores += self.item_bias
        # [bs * max_pred_num, word_num]
        pred_prob = nn.Softmax(dim=-1)(pred_scores)
        # [bs * max_pred_num]
        per_example_loss = -pred_prob.gather(dim=1, index=label_ids.view([-1, 1])).reshape([-1])
        loss = per_example_loss.dot(label_weights) / torch.sum(label_weights)

        return loss


class BertForSeqRec(BertModel):
    """BERT for Sequential Recommendation
    """

    def __init__(self, config):
        super(BertForSeqRec, self).__init__(config)
        self.bert = BertModel(config)
        self.predictor = BertPredictor(config)
        self.lossFunc = LossForBERT(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, masked_lm_positions=None,
                masked_lm_labels=None, masked_lm_weights=None, negative_ids_list=None):
        """
        input_ids: [bs, max_seq_len]
        attention_masks: [bs, max_seq_len]
        masked_positions: [bs, max_pred_num]
        target_ids: [bs * max_pred_num]
        masked_lm_labels: [bs * max_pred_num]
        negative_ids_list: [neg_num, bs, max_pred_num]
        """
        bert_context = self.bert(input_ids, None, attention_mask)
        pred_context = self.predictor(bert_context, masked_positions=masked_lm_positions)
        loss = self.lossFunc(pred_context=pred_context,
                             label_ids=masked_lm_labels,
                             negative_ids_list=negative_ids_list,
                             label_weights=masked_lm_weights)
        return loss

    def forward_both_raw_and_aug(self, raw_batch, aug_labels):
        """
        raw_batch:
            input_ids: [bs, max_seq_len]
            attention_masks: [bs, max_seq_len]
            masked_positions: [bs, max_pred_num]
            target_ids: [bs * max_pred_num]
            masked_lm_labels: [bs * max_pred_num]
            negative_ids_list: [neg_num, bs, max_pred_num]
        aug_labels:
            pos_augs: [bs * max_pred_num, word_num], one-hot
            neg_augs: [bs * max_pred_num, word_num], one-hot
            aug_weights: [bs * max_pred_num], prob dist
        """
        input_ids, attention_masks, masked_positions, \
        masked_ids, label_weights, negative_ids_list = raw_batch

        bert_context = self.bert(input_ids, None, attention_masks)
        pred_context = self.predictor(bert_context, masked_positions=masked_positions)

        raw_loss = self.lossFunc(pred_context=pred_context,
                                 label_ids=masked_ids,
                                 negative_ids_list=negative_ids_list,
                                 label_weights=label_weights)

        pos_augs, neg_augs, aug_weights = aug_labels
        aug_loss = self.lossFunc.get_aug_loss(pred_context, pos_augs, neg_augs, aug_weights)

        return raw_loss, aug_loss

    def get_train_scores(self, input_ids, attention_mask, masked_lm_positions):
        """
        input_ids: [bs, max_seq_len]
        attention_masks: [bs, max_seq_len]
        masked_positions: [bs, max_pred_num]
        """
        bert_context = self.bert(input_ids, None, attention_mask)
        # [bs * max_pred_num, hidden_size]
        pred_context = self.predictor(bert_context, masked_positions=masked_lm_positions)

        word_weights = self.lossFunc.word_weights
        # if self.lossFunc.loss_type is not 'full_cross_entropy':
        word_weights = word_weights.transpose(0, 1)
        # [bs * max_pred_num, word_num] <- [bs * max_pred_num, hidden_size] * [hidden_size, word_num]
        pred_scores = torch.mm(pred_context, word_weights)
        if self.lossFunc.item_bias is not None:
            pred_scores += self.lossFunc.item_bias
        # [bs * max_pred_num, word_num]
        # pred_prob = nn.Softmax(dim=-1)(pred_scores)

        return pred_scores

    def pred_test_scores(self, input_ids,
                         attention_masks,
                         masked_positions,
                         masked_ids):
        """
        test_input_ids: [bs, max_seq_len]
        test_attention_masks: [bs, max_seq_len]
        test_masked_positions: [bs, 1]
        target_ids: [bs, (1 + eval_neg_num)]
        """
        bs = input_ids.size(0)
        # [bs * (1 + eval_neg_num)]
        target_ids = masked_ids.reshape([-1])
        # [bs * (1 + eval_neg_num)]
        bert_context = self.bert(input_ids, None, attention_masks)
        # [bs, hidden_size]
        pred_context = self.predictor(bert_context, masked_positions=masked_positions)
        hidden_size = pred_context.size(1)
        # [bs, (1 + eval_neg_num), hidden_size]
        target_embeds = self.bert.embeddings.word_embeddings.weight.index_select(dim=0, index=target_ids) \
            .reshape([bs, -1, hidden_size])
        # [bs, 1, hidden_size]
        pred_context.unsqueeze_(dim=1)
        # [bs, 1, hidden_size] * [bs, (1 + eval_neg_num), hidden_size]
        # -> [bs, (1 + eval_neg_num), hidden_size] -> [bs, (1 + eval_neg_num)]
        pred_scores = torch.mul(pred_context, target_embeds).sum(dim=2, keepdim=False)
        if self.lossFunc.item_bias is not None:
            # [bs, (1 + eval_neg_num)]
            target_bias = torch.index_select(self.lossFunc.item_bias, dim=0, index=target_ids).reshape([bs, -1])
            pred_scores += target_bias
        # [bs, (1 + eval_neg_num)]
        return pred_scores

    def evaluate(self,
                 test_input_ids,
                 test_attention_masks,
                 test_masked_positions,
                 target_ids):
        """
        test_input_ids: [bs, max_seq_len]
        test_attention_masks: [bs, max_seq_len]
        test_masked_positions: [bs, 1]
        target_ids: [bs, (1 + eval_neg_num)]
        """

        ranks = self.pred_test_scores(test_input_ids, test_attention_masks, test_masked_positions, target_ids).argsort(
            dim=1, descending=True).argsort(dim=1, descending=False)[:, 0:1].float()

        # evaluate ranking
        metrics = {
            'ndcg_1': 0,
            'ndcg_5': 0,
            'ndcg_10': 0,
            'ndcg_20': 0,
            'hit_1': 0,
            'hit_5': 0,
            'hit_10': 0,
            'hit_20': 0,
            'ap': 0,
        }
        for rank in ranks:
            if rank < 1:
                metrics['ndcg_1'] += 1
                metrics['hit_1'] += 1
            if rank < 5:
                metrics['ndcg_5'] += 1 / torch.log2(rank + 2)
                metrics['hit_5'] += 1
            if rank < 10:
                metrics['ndcg_10'] += 1 / torch.log2(rank + 2)
                metrics['hit_10'] += 1
            if rank < 20:
                metrics['ndcg_20'] += 1 / torch.log2(rank + 2)
                metrics['hit_20'] += 1
            metrics['ap'] += 1.0 / (rank + 1)
        return metrics


def fit_batch(train_batch, model, optimizer):
    input_ids, attention_masks, masked_positions, \
    masked_ids, label_weights, negative_ids_list = train_batch

    model.train()
    loss = model(input_ids=input_ids, attention_mask=attention_masks, masked_lm_positions=masked_positions,
                 masked_lm_labels=masked_ids, masked_lm_weights=label_weights, negative_ids_list=negative_ids_list)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

import torch
from torch import nn
import math
import torch.nn.functional as F
from model_utils.EMA import EMA


class ContextEncoder(nn.Module):
    def __init__(self, config, graph=None, user_embed=None, item_embed=None):
        super().__init__()
        self.config = config
        self.device = config['device']
        if user_embed is None:
            self.user_embeddings = nn.Embedding(config['user_num'], config['hidden_size'])
            self.item_embeddings = nn.Embedding(config['item_num'], config['hidden_size'], padding_idx=0)
        else:
            self.user_embeddings = user_embed
            self.item_embeddings = item_embed

        self.graph = graph

    def forward(self, user_id, hist_item_ids, rectify=True):
        raise NotImplementedError

    def obtain_embeds(self):
        users_emb = self.user_embeddings.weight
        if self.graph is None:
            items_emb = self.item_embeddings.weight
        else:
            items_emb = self.item_embeddings.weight
            all_emb = items_emb
            # [numItem, hidden_size]
            embs = [all_emb]
            g_droped = self.graph
            for layer in range(self.config['n_layers']):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            # [numItem, hidden_size, n_layers + 1]
            embs = torch.stack(embs, dim=2)
            # [numItem, hidden_size]
            items_emb = torch.mean(embs, dim=2)

        return users_emb, items_emb


class BirDRec(nn.Module):
    def __init__(self, config, forward_encoder: ContextEncoder, backward_encoder: ContextEncoder = None):
        super().__init__()
        self.config = config
        self.device = config['device']

        self.item_bias = nn.Embedding(config['item_num'], 1, padding_idx=0)

        if self.config['train_type'] == 'train':
            self.forward_encoder = forward_encoder
            # if self.config['corr_input']:
            #     self.backward_encoder = backward_encoder
            self.apply(self._init_parameters)

        self.corr_threshold = config['corr_threshold']
        self.corr_epoch = config['corr_epoch']
        self.rectify_target = self.config['rectify_target']
        self.rectify_input = self.config['rectify_input']
        self.replace_target = self.config['replace_target']
        self.delete_target_ratio_upper = config['delete_target_ratio']
        self.delete_input_ratio_upper = config['delete_input_ratio']
        self.replace_target_ratio_upper = config['replace_ratio']
        # self.corr_type = self.config['corr_type']
        self.temperature = config['temperature']
        self.train_neg_num = 250
        self.target_neg_num = 8
        self.weighted_loss = self.config['weighted_loss']

        self.moving_avg_model = None
        self.self_ensemble = self.config['self_ensemble']

    def set_encoder(self, encoder):
        self.forward_encoder = encoder

    def _init_parameters(self, module):
        """ Initialize the parameterss.
        """
        if isinstance(module, nn.Embedding):
            hidden_size = module.weight.size()[1]
            bound = 6 / math.sqrt(hidden_size)
            nn.init.uniform_(module.weight, a=-bound, b=bound)
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, train_batch, epoch_num):
        user_id, hist_item_ids, pos_target, replace_candidates, sample_idices = train_batch
        cand_size = replace_candidates.shape[1]
        sampled_neg_targets = torch.randint(low=1, high=self.config['item_num'], size=(user_id.shape[0], 250)).type(torch.long)
        if torch.cuda.is_available():
            user_id = user_id.to(self.device)
            hist_item_ids = hist_item_ids.to(self.device)
            pos_target = pos_target.to(self.device)
            replace_candidates = replace_candidates.to(self.device)
            sampled_neg_targets = sampled_neg_targets.to(self.device)
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        pos_target = [bs]
        neg_targets = [bs, neg_num]
        """

        rectified_input_ratio = 0
        rectified_target_ratio = 0

        if epoch_num >= self.corr_epoch:
            if self.rectify_input or self.rectify_target:
                if self.self_ensemble and self.moving_avg_model is None:
                    self.moving_avg_model = EMA(self.forward_encoder)
                    self.moving_avg_model.model.eval()

        if epoch_num >= self.corr_epoch:
            if self.rectify_input:
                pos_score_last, pos_score_second_last, \
                    candidate_scores_last, candidate_scores_second_last, \
                    neg_scores_last, neg_scores_second_last \
                    = self.get_various_scores_last_and_second_last(self.forward_encoder, user_id, hist_item_ids,
                                                                   pos_target,
                                                                   replace_candidates, sampled_neg_targets)
                # [bs, 1]
                rectified_ema_pos_score = None
                rectified_ema_neg_scores = None
                if self.self_ensemble: # rectify input with ensemble
                    ema_pos_score_last, ema_pos_score_second_last, \
                        _, _, \
                        ema_neg_scores_last, ema_neg_scores_second_last = \
                        self.get_various_scores_last_and_second_last(self.moving_avg_model.model, user_id, hist_item_ids, pos_target,
                                                     replace_candidates, sampled_neg_targets)
                    input_select_condition, rectified_input_ratio = self.get_rectify_input_conditions(ema_pos_score_last,
                                                                                                      ema_neg_scores_last)
                    rectified_ema_pos_score = torch.where(input_select_condition > 0, ema_pos_score_last, pos_score_second_last)
                    rectified_ema_neg_scores = torch.where(input_select_condition > 0, ema_neg_scores_last, ema_neg_scores_second_last)
                else:  # rectify input without ensemble
                    input_select_condition, rectified_input_ratio = self.get_rectify_input_conditions(pos_score_last, neg_scores_last)

                # rectify pos, neg, and cand scores according to the condition
                rectified_pos_score = torch.where(input_select_condition > 0, pos_score_last, pos_score_second_last)
                rectified_neg_score = torch.where(input_select_condition > 0, neg_scores_last, neg_scores_second_last)
                rectified_cand_score = torch.where(input_select_condition > 0, candidate_scores_last, candidate_scores_second_last)

                if self.rectify_target:  # rectify input and rectify target
                    if self.self_ensemble: # rectify input and rectify target, with ensemble
                        loss, rectified_target_ratio, update_pool_indices = self.correct_target_with_ensemble(
                            rectified_pos_score, rectified_neg_score, rectified_cand_score, rectified_ema_pos_score, rectified_ema_neg_scores)
                    else:  # rectify input and rectify target, without ensemble
                        loss, rectified_target_ratio, update_pool_indices = self.correct_target_no_ensemble(
                            rectified_pos_score, rectified_neg_score, rectified_cand_score)
                else:  # rectify input, do not rectify target
                    loss = -(torch.log(torch.sigmoid(rectified_pos_score - rectified_neg_score[:, cand_size:cand_size + 2])))
                # calcualte loss
                loss = loss.sum(dim=1, keepdim=True)
                return loss, sample_idices, rectified_input_ratio, rectified_target_ratio, update_pool_indices

            else:  # do not rectify input
                pos_score, candidate_scores, neg_scores = \
                    self.get_various_scores_last(self.forward_encoder, user_id, hist_item_ids, pos_target,
                                                 replace_candidates, sampled_neg_targets)
                update_pool_indices = None
                if self.rectify_target:  # do not rectify input, rectify target
                    if self.self_ensemble: # do not rectify input, rectify target, with ensemble
                        ema_pos_score_last, _, _, _, ema_neg_scores_last, _ = \
                            self.get_various_scores_last_and_second_last(self.moving_avg_model.model, user_id, hist_item_ids,
                                                                         pos_target,
                                                                         replace_candidates, sampled_neg_targets)
                        loss, rectified_target_ratio, update_pool_indices = \
                            self.correct_target_with_ensemble(pos_score, neg_scores, candidate_scores, ema_pos_score_last, ema_neg_scores_last)
                    else:  # do not rectify input, rectify target, without ensemble
                        loss, rectified_target_ratio, update_pool_indices = self.correct_target_no_ensemble(pos_score,
                                                                                                            neg_scores,
                                                                                                            candidate_scores)
                else: # do not rectify input, do not rectify target
                    loss = -(torch.log(torch.sigmoid(pos_score - neg_scores[:, cand_size:cand_size + 2])))

                loss = loss.sum(dim=1, keepdim=True)

                return loss, sample_idices, rectified_input_ratio, rectified_target_ratio, update_pool_indices
        else:  # epoch < self.corr_epoch:
            pos_score, candidate_scores, neg_scores = \
                self.get_various_scores_last(self.forward_encoder, user_id, hist_item_ids, pos_target,
                                             replace_candidates, sampled_neg_targets)
            loss = -(torch.log(torch.sigmoid(pos_score - neg_scores[:, cand_size:cand_size + 2])))
            loss = loss.sum(dim=1, keepdim=True)
            return loss, sample_idices, rectified_input_ratio, rectified_target_ratio, None

    def get_various_scores_last(self, model, user_id, hist_item_ids, pos_target, replace_candidates, sampled_neg_targets):
        _, item_embeds = model.obtain_embeds()
        # neg_num = neg_targets.size()[1]
        # [bs, cand_szie + neg_num]
        candidates_and_neg_items = torch.cat([replace_candidates, sampled_neg_targets], dim=1)
        # [bs, cand_szie + neg_num, hidden_size]
        candidates_and_neg_embeds = item_embeds[candidates_and_neg_items]
        # [bs, hidden_size]
        pos_target_embeds = item_embeds[pos_target]

        context_embed = model(user_id, hist_item_ids, rectify=False)
        # [bs, 1]
        pos_score = torch.mul(context_embed, pos_target_embeds).sum(dim=1, keepdim=True)
        # [bs, cand_szie + neg_num]
        candidates_and_neg_scores = torch.mul(context_embed.unsqueeze(1), candidates_and_neg_embeds).sum(dim=2,
                                                                                                         keepdim=False)
        # [bs, cand_szie]
        candidate_scores = candidates_and_neg_scores[:, 0:replace_candidates.shape[1]]
        # [bs, neg_num]
        neg_scores = candidates_and_neg_scores[:, replace_candidates.shape[1]:]

        return pos_score, candidate_scores, neg_scores

    def get_various_scores_last_and_second_last(self, model, user_id, hist_item_ids, pos_target, replace_candidates, sampled_neg_targets):
        _, item_embeds = model.obtain_embeds()
        # neg_num = neg_targets.size()[1]
        # [bs, cand_szie + neg_num]
        candidates_and_neg_items = torch.cat([replace_candidates, sampled_neg_targets], dim=1)
        # [bs, cand_szie + neg_num, hidden_size]
        candidates_and_neg_embeds = item_embeds[candidates_and_neg_items]
        # [bs, hidden_size]
        pos_target_embeds = item_embeds[pos_target]

        # [bs, hidden_size]
        context_embed_last, context_embed_second_last = model(user_id, hist_item_ids, rectify=True)
        # [bs, 1]
        pos_score_last = torch.mul(context_embed_last, pos_target_embeds).sum(dim=1, keepdim=True)
        pos_score_second_last = torch.mul(context_embed_second_last, pos_target_embeds).sum(dim=1, keepdim=True)
        # [bs, cand_szie + neg_num]
        candidates_and_neg_scores_last = torch.mul(context_embed_last.unsqueeze(1), candidates_and_neg_embeds).sum(
            dim=2, keepdim=False)
        candidates_and_neg_scores_second_last = torch.mul(context_embed_second_last.unsqueeze(1),
                                                          candidates_and_neg_embeds).sum(dim=2, keepdim=False)
        # [bs, cand_szie]
        candidate_scores_last = candidates_and_neg_scores_last[:, 0:replace_candidates.shape[1]]
        candidate_scores_second_last = candidates_and_neg_scores_second_last[:, 0:replace_candidates.shape[1]]
        # [bs, neg_num]
        neg_scores_last = candidates_and_neg_scores_last[:, replace_candidates.shape[1]:]
        neg_scores_second_last = candidates_and_neg_scores_second_last[:, replace_candidates.shape[1]:]

        return pos_score_last, pos_score_second_last, \
            candidate_scores_last, candidate_scores_second_last, \
            neg_scores_last, neg_scores_second_last

    def get_rectify_input_conditions(self, pos_score_last, neg_scores_last):
        sorted_neg_scores_last, sorted_neg_indices_last = torch.sort(neg_scores_last, descending=True, dim=1)
        delete_threshold_position = int(self.train_neg_num * self.delete_input_ratio_upper)
        delete_judger_score_last = sorted_neg_scores_last[:,
                                   delete_threshold_position - 1: delete_threshold_position]
        select_condition = pos_score_last - delete_judger_score_last
        input_delete_mask = (select_condition > 0).long().to(self.device)
        flipped_input_delete_mask = 1 - input_delete_mask
        rectified_input_ratio = (flipped_input_delete_mask.sum() / flipped_input_delete_mask.shape[0])

        return select_condition, torch.sum(rectified_input_ratio)

    def correct_target_no_ensemble(self, pos_score, neg_scores, cand_scores):
        """
        :param pos_score: [bs, 1]
        :param neg_scores: [bs, cand_size + train_neg_num]
        :param candidates_and_neg_items: [bs, cand_size + train_neg_num]
        # :param candidate_scores: [bs, cand_size]
        # :param full_neg_scores: [bs, train_neg_num]
        # :param target_neg_scores: [bs, target_neg_num]
        :param type: 0 remove target, 1 replace target
        :return:
        """
        cand_size = cand_scores.shape[1]
        # [bs, cand_size + train_neg_num]
        sorted_neg_scores, sorted_neg_indices = torch.sort(neg_scores, descending=True, dim=1)
        delete_threshold_position = int(self.train_neg_num * self.delete_target_ratio_upper)

        update_pool_indices = None
        if self.config['replace_target']:
            # replace low-scored targets
            sorted_cand_scores, sorted_cand_indices = torch.sort(cand_scores, descending=True, dim=1)
            replace_threshold_position = int(self.train_neg_num * self.replace_target_ratio_upper)
            replace_judger_score = sorted_neg_scores[:, replace_threshold_position - 1: replace_threshold_position]
            rectify_condition = pos_score - replace_judger_score > 0
            max_cand_score = sorted_cand_scores[:, 0:1]
            pos_score = torch.where(rectify_condition, pos_score, max_cand_score)
            # get indices to update the pool
            # update_condition = max_cand_score - sorted_neg_scores[:, cand_size-1: cand_size] > 0
            # old_max_item = torch.gather(input=replace_cands, dim=1, index=sorted_cand_indices[:, 0: 1])
            # new_items = torch.gather(input=sampled_negs, dim=1, index=sorted_neg_indices[:, 0: cand_size-1])
            # items_for_update = torch.concat([old_max_item, new_items], dim=1)
            # update_pool_indices = torch.where(update_condition, replace_cands, items_for_update)

        delete_judger_score = sorted_neg_scores[:, delete_threshold_position - 1: delete_threshold_position]
        target_neg_scores = sorted_neg_scores[:, 0: int(self.target_neg_num)]
        instance_delete_mask = (pos_score > delete_judger_score).long().to(self.device)
        rectified_target_ratio = 1 - (instance_delete_mask.sum() / instance_delete_mask.shape[0])
        raw_loss = self.bpr_loss(pos_score, target_neg_scores, weighted=False)
        corrected_loss = torch.mul(instance_delete_mask, raw_loss)

        return corrected_loss, rectified_target_ratio, update_pool_indices

    def correct_target_with_ensemble(self, pos_score, neg_scores, cand_scores, ema_pos_score, ema_neg_scores):
        """
        :param pos_score: [bs, 1]
        :param neg_scores: [bs, cand_size + train_neg_num]
        :param candidates_and_neg_items: [bs, cand_size + train_neg_num]
        # :param candidate_scores: [bs, cand_size]
        # :param full_neg_scores: [bs, train_neg_num]
        # :param target_neg_scores: [bs, target_neg_num]
        :param type: 0 remove target, 1 replace target
        :return:
        """
        cand_size = cand_scores.shape[1]
        # [bs, cand_size + train_neg_num]
        sorted_neg_scores, _ = torch.sort(neg_scores, descending=True, dim=1)
        ema_sorted_neg_scores, _ = torch.sort(ema_neg_scores, descending=True, dim=1)
        delete_threshold_position = int(self.train_neg_num * self.delete_target_ratio_upper)

        update_pool_indices = None
        if self.config['replace_target']:
            # replace low-scored targets
            sorted_cand_scores, sorted_cand_indices = torch.sort(cand_scores, descending=True, dim=1)
            replace_threshold_position = int(self.train_neg_num * self.replace_target_ratio_upper)
            replace_judger_score = ema_sorted_neg_scores[:, replace_threshold_position - 1: replace_threshold_position]
            rectify_condition = ema_pos_score - replace_judger_score > 0
            max_cand_score = sorted_cand_scores[:, 0:1]
            pos_score = torch.where(rectify_condition, pos_score, max_cand_score)
            # get indices to update the pool
            # update_condition = max_cand_score - sorted_neg_scores[:, cand_size-1: cand_size] > 0
            # old_max_item = torch.gather(input=replace_cands, dim=1, index=sorted_cand_indices[:, 0: 1])
            # new_items = torch.gather(input=sampled_negs, dim=1, index=sorted_neg_indices[:, 0: cand_size-1])
            # items_for_update = torch.concat([old_max_item, new_items], dim=1)
            # update_pool_indices = torch.where(update_condition, replace_cands, items_for_update)

        delete_judger_score = ema_sorted_neg_scores[:, delete_threshold_position - 1: delete_threshold_position]
        target_neg_scores = sorted_neg_scores[:, 0: int(self.target_neg_num)]
        instance_delete_mask = (ema_pos_score > delete_judger_score).long().to(self.device)
        rectified_target_ratio = 1 - (instance_delete_mask.sum() / instance_delete_mask.shape[0])
        raw_loss = self.bpr_loss(pos_score, target_neg_scores, weighted=False)
        corrected_loss = torch.mul(instance_delete_mask, raw_loss)

        return corrected_loss, rectified_target_ratio, update_pool_indices

    def bpr_loss(self, pos_score, neg_scores, weighted=False):
        """
        :param pos_score: [bs, 1]
        :param neg_scores: [bs, target_train_num] for weighted, [bs, target_neg_num] for not weighted
        :param weighted: True for full negative, false for high-rank negative
        :return:
        """
        if weighted:
            pred = pos_score - neg_scores
            Z = torch.negative(F.logsigmoid(pred))
            tau = torch.sqrt(torch.var(Z, dim=1, keepdim=True) / self.target_neg_num)
            importance = F.softmax(Z / tau, dim=1)
            return importance.detach() * Z
        else:
            return -(torch.log(torch.sigmoid(pos_score - neg_scores)))

    def eval_ranking(self, test_batch, model=0):
        user_id, hist_item_ids, target_ids, pad_indices = test_batch
        if torch.cuda.is_available():
            user_id = user_id.to(torch.device('cuda'))
            hist_item_ids = hist_item_ids.to(torch.device('cuda'))
            target_ids = target_ids.to(torch.device('cuda'))
            pad_indices = pad_indices.to(torch.device('cuda')).double()
            # print(f'eval_ranking: {user_id.is_cuda}')
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        masks = [bs, seq_len]
        target_ids = [bs, eval_neg_num + 2]
        """
        # [bs, pred_num]
        if model == 0:
            scores = self._get_test_scores(self.forward_encoder, user_id, hist_item_ids, target_ids) + pad_indices
        else:
            scores = self._get_test_scores(self.moving_avg_model.model, user_id, hist_item_ids, target_ids) + pad_indices
        pos_score = scores[:, 0: 1]
        neg_scores = scores[:, 1: -1]
        ranks = (neg_scores > pos_score).long().sum(dim=1, keepdim=False)
        # ranks = scores.argsort(dim=1, descending=True).argsort(dim=1, descending=False)[:, 0:1].float()

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

    def _get_test_scores(self, model, user_id, hist_item_ids, target_ids):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        masks = [bs, seq_len]
        target_ids = [bs, 1 + eval_neg_num]
        """
        user_embeds, item_embeds = model.obtain_embeds()
        # [bs, 1, hidden_size]
        context_embed = model(user_id, hist_item_ids, rectify=False).unsqueeze(1)
        # [bs, pred_num, hidden_size]
        target_embeds = item_embeds[target_ids]
        # [bs, pred_num]
        scores = torch.mul(context_embed, target_embeds).sum(dim=2, keepdim=False)

        return scores







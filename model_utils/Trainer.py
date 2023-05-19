from data_utils.RankingEvaluator import RankingEvaluator
import torch
from torch import optim
import time
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.BirDRec import BirDRec
from model.SASRec import SASRec
from model.Caser import BERD_Caser
from model.GRU4Rec import BERD_GRU4Rec
from model.FPMC import FPMC
from model.FMLPRec import FMLPRec
from model.BERT4Rec import BERT4Rec
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class Trainer:
    def __init__(self, config, data_model, save_dir):

        self.config = config
        self.save_dir = save_dir
        self.train_type = config['train_type']
        self.rec_model = config['rec_model']
        self.model_save_dir = './datasets/' + self.config['dataset'] + '/model/'
        self.model_save_path = self.model_save_dir + self.rec_model + str(self.config['sample_loss_weight']) + '-'
        self.data_analysis_dir = './datasets/' + self.config['dataset'] + '/analysis/'
        self.save_epochs = self.config['save_epochs']

        if self.train_type == 'analysis':
            train_dataset, train_loader = data_model.generate_train_dataloader_unidirect()
            test_loader = None
        else:
            train_dataset, train_loader = data_model.generate_train_dataloader_unidirect()
            test_loader = data_model.generate_test_dataloader_unidirect()
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self._evaluator = RankingEvaluator(test_loader)
        self.train_size = len(train_loader.dataset)
        graph = None
        if self.config['rec_model'] == 'MAGNN':
            graph = data_model.getSparseGraph()
        seq_encoder = self.getSeqEncoder(graph)

        rec_model = BirDRec(config, seq_encoder)

        if self.train_type == 'train':
            if rec_model is not None:
                self._model = rec_model
                self._device = config['device']
                self._model.double().to(self._device)
                self._optimizer = _get_optimizer(
                    self._model.forward_encoder, learning_rate=config['learning_rate'],
                    weight_decay=config['weight_decay'])
                self.scheduler = ReduceLROnPlateau(self._optimizer, 'max', patience=10,
                                                   factor=config['decay_factor'])
            self.forget_rates = self.build_forget_rates()
        else:
            self._device = config['device']
            self._model = rec_model



    def getSeqEncoder(self, graph):
        if self.config['rec_model'] == 'SASRec':
            return SASRec(self.config, graph)
        elif self.config['rec_model'] == 'GRU4Rec':
            return BERD_GRU4Rec(self.config, graph)
        elif self.config['rec_model'] == 'Caser':
            return BERD_Caser(self.config, graph)
        elif self.config['rec_model'] == 'FPMC':
            return FPMC(self.config, graph)
        elif self.config['rec_model'] == 'FMLPRec':
            return FMLPRec(self.config, graph)
        elif self.config['rec_model'] == 'BERT4Rec':
            return BERT4Rec(self.config, graph)

    def build_forget_rates(self):
        forget_rates = np.ones(self.config['epoch_num']) * 0.06
        forget_rates[:20] = np.linspace(0, 0.06, 20)
        return forget_rates

    def save_model(self, epoch_num):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        corr_str = ''
        if self.config['rectify_input']:
            corr_str += 'rectify_input'
        if self.config['rectify_target']:
            corr_str += 'rectify_target'
        save_path = self.model_save_path + str(epoch_num) + corr_str + '-model.pkl'
        torch.save(self._model.forward_encoder, save_path)
        print(f'model saved at {save_path}')

    def load_model(self, epoch_num):
        corr_str = ''
        if self.config['rectify_input']:
            corr_str += 'rectify_input'
        if self.config['rectify_target']:
            corr_str += 'rectify_target'
        load_path = self.model_save_path + str(epoch_num) + corr_str + '-model.pkl'
        print(f'loading model from {load_path}')

        if torch.cuda.is_available():
            model = torch.load(load_path)
        else:
            model = torch.load(load_path, map_location=torch.device('cpu'))

        return model

    def train_one_batch(self, batch, epoch_num):
        self._model.forward_encoder.train()
        self._optimizer.zero_grad()

        # [bs], [bs], [1]
        loss, sample_idices, rectified_input_ratio, rectified_target_ratio, update_pool_indices = self._model(batch, epoch_num)
        overall_loss = loss.sum()
        overall_loss.backward()

        self._optimizer.step()

        if self._model.moving_avg_model is None:
            pass
        else:
            self._model.moving_avg_model.update_params(self._model.forward_encoder)
            self._model.moving_avg_model.apply_shadow()

        if update_pool_indices is None:
            pass
        else:
            self.train_dataset.update_replacement_candidate_pool(sample_idices, update_pool_indices)

        return overall_loss, rectified_input_ratio, rectified_target_ratio

    def train(self):
        if self.train_type == 'train':
            print('=' * 60, '\n', 'Start Training', '\n', '=' * 60, sep='')
            keep_train = True
            for epoch in range(self.config['epoch_num']):
                start_train = time.time()
                loss_iter = 0
                rectified_input_ratios = 0
                rectified_target_ratios = 0
                for batch in self.train_loader:
                    loss, rectified_input_ratio, rectified_target_ratio = self.train_one_batch(batch, epoch)
                    loss_iter += loss.item()
                    rectified_input_ratios += rectified_input_ratio
                    rectified_target_ratios += rectified_target_ratio
                len_train_loader = len(self.train_loader)

                print(f'################## epoch {epoch} ###########################')
                print(
                    f"loss: {round(loss_iter / len(self.train_loader), 4)}, len_train_loader:{len_train_loader}, train time: {time.time() - start_train}")
                print(f"rectified_input_ratio: {rectified_input_ratios / len_train_loader}")
                print(f"rectified_target_ratio: {rectified_target_ratios / len_train_loader}")
                start_eval = time.time()
                keep_train = self.evaluate(epoch)
                print(f"eval time: {time.time() - start_eval}")
                print('#########################################################')
                if epoch in self.save_epochs:
                    self.save_model(epoch)
                if not keep_train:
                    break
        elif self.train_type == 'eval':
            for epoch in self.save_epochs:
                self._model.set_encoder(self.load_model(epoch))
                self._model.double().to(self._device)
                self._evaluator.evaluate(model=self._model, train_iter=epoch)

    def data_analysis_new(self):

        prob_gap = np.load(self.data_analysis_dir + 'prob_gap.npy')

        self.draw_fig_new(prob_gap)

    def draw_fig_new(self, prob_gap):
        # generate data points
        alpha = 0.05
        step = 0.005
        x = []
        y = []
        while alpha < 0.9:
            alpha += step
            prob_less_than_alpha = len(prob_gap[prob_gap < alpha]) / prob_gap.shape[0]
            if prob_less_than_alpha < 1e-6:
                continue
            x.append(np.log2(alpha))
            y.append(np.log2(prob_less_than_alpha + 1e-8))
        x = np.array(x)
        y = np.array(y)
        res = stats.linregress(x, y)

        estimated_c = round(np.exp2(res.intercept), 4)
        estimated_lambda = round(res.slope, 4)

        font = {
                'weight': 'normal',
                'size': 20,
                }
        sns.set_style("whitegrid")
        blue, = sns.color_palette("muted", 1)

        plt.plot(x, y, 'o', color='lightblue', label=r'Observed (log($\alpha$), log($F_{\alpha}$))', markeredgecolor=blue, markeredgewidth=0.5)
        plt.plot(x, res.intercept + res.slope * x, 'indianred', linewidth=4, alpha=0.8, label=r'Regression Line: $\lambda$log($\alpha$)+log$(C)$')

        x_position = (x.max() - x.min()) * 0.5 + x.min()
        y_position = (y.max() - y.min()) * 0.02 + y.min()
        t = r'Estimated $C$: {:.4f}'.format(estimated_c) + '\n' + r'Estimated $\lambda$: {:.4f}'.format(
            estimated_lambda)

        # plt.figure(figsize=(4, 3), dpi=150)
        plt.text(x=x_position, y=y_position, s=t, bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), fontsize=16)
        # plt.text(x=0.9, y=1.4, s=r'Estimated $\lambda$: {:.4f}'.format(estimated_lambda),
        #          bbox=dict(facecolor='grey', alpha=0.1))
        plt.xlabel(r'log($\alpha$)', fontdict=font)
        plt.ylabel(r'log($F_{\alpha}$)', fontdict=font)

        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.xticks(np.arange(-4, 1, step=1))

        ax = plt.gca()
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)

        plt.grid(linestyle='--', linewidth=1)

        plt.legend(handlelength=0.5, fontsize=13.5)

        plt.savefig(self.data_analysis_dir + self.config['dataset'] + f'-{self.rec_model}.png', dpi=300)
        plt.show()

    def draw_fig(self, max_sum_num, mid_sum_num, all_pred_scores):
        max_value = np.sum(all_pred_scores[:, -max_sum_num:], axis=1, keepdims=False)
        # [bs, 1]
        # beauty 240,
        mid_value = np.sum(all_pred_scores[:, 0:mid_sum_num], axis=1, keepdims=False)
        # [bs, 1]
        prob_gap = max_value - mid_value
        # [bs]
        np.save(self.data_analysis_dir + 'prob_gap', prob_gap)

        # generate data points
        alpha = 0.05
        step = 0.005
        x = []
        y = []
        while alpha < 0.9:
            alpha += step
            prob_less_than_alpha = len(prob_gap[prob_gap < alpha]) / prob_gap.shape[0]
            if prob_less_than_alpha < 1e-6:
                continue
            x.append(np.log2(alpha))
            y.append(np.log2(prob_less_than_alpha + 1e-8))
        x = np.array(x)
        y = np.array(y)
        res = stats.linregress(x, y)

        estimated_c = round(np.exp2(res.intercept), 4)
        estimated_lambda = round(res.slope, 4)

        font = {
                'weight': 'normal',
                'size': 20,
                }
        sns.set_style("whitegrid")
        blue, = sns.color_palette("muted", 1)

        plt.plot(x, y, 'o', color='lightblue', label=r'Observed (log($\alpha$), log($F_{\alpha}$))', markeredgecolor=blue, markeredgewidth=0.5)
        plt.plot(x, res.intercept + res.slope * x, 'indianred', linewidth=4, alpha=0.8, label=r'Regression Line: $\lambda$log($\alpha$)+log$(C)$')

        x_position = (x.max() - x.min()) * 0.5 + x.min()
        y_position = (y.max() - y.min()) * 0.02 + y.min()
        t = r'Estimated $C$: {:.4f}'.format(estimated_c) + '\n' + r'Estimated $\lambda$: {:.4f}'.format(
            estimated_lambda)

        # plt.figure(figsize=(4, 3), dpi=150)
        plt.text(x=x_position, y=y_position, s=t, bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), fontsize=16)
        # plt.text(x=0.9, y=1.4, s=r'Estimated $\lambda$: {:.4f}'.format(estimated_lambda),
        #          bbox=dict(facecolor='grey', alpha=0.1))
        plt.xlabel(r'log($\alpha$)', fontdict=font)
        plt.ylabel(r'log($F_{\alpha}$)', fontdict=font)

        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.xticks(np.arange(-4, 1, step=1))

        ax = plt.gca()
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)

        plt.grid(linestyle='--', linewidth=1)

        plt.legend(handlelength=0.5, fontsize=13.5)

        plt.savefig(self.data_analysis_dir + f'{max_sum_num}={mid_sum_num}-{self.rec_model}.png', dpi=300)
        plt.show()

    def evaluate(self, iter):
        self._model.eval()
        if self._model.moving_avg_model is None:
            keep_train, ndcg10 = self._evaluator.evaluate(model=self._model, train_iter=iter, eval_model=0)
            self.scheduler.step(ndcg10)
        else:
            keep_train, ndcg10 = self._evaluator.evaluate(model=self._model, train_iter=iter, eval_model=1)
        return keep_train

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer



def _get_optimizer(model, learning_rate, weight_decay=0.01):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)


def set2str(input_set):
    set_str = ''
    set_len = len(input_set)
    for i, item in enumerate(input_set):
        if i < set_len - 1:
            set_str += str(item) + ','
        else:
            set_str += str(item)
    return set_str


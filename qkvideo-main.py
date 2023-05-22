import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import torch

from model_utils.Trainer import Trainer
from data_utils.GraphDataGenerator import GraphDataCollector
from data_utils.RankingEvaluator import print_dict
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    config = {
        # data settings
        'thread_num': 4,
        'dataset': 'yelp',  # electronics
        'eval_neg_num': 'full',
        'input_len': 5,
        'threshold': 1,
        # training settings
        'rec_model': 'SASRec',
        'train_type': 'train',
        'co-train': False,
        'filter-with-loss': False,
        'save_epochs': [300, 400],
        'epoch_num': 500,
        'learning_rate': 0.001,   # 0.01 for yelp
        'train_batch_size': 1024,
        'test_batch_size': 512,
        'drop_ratio': 0.1,
        # graph settings
        'n_layers': 2,
        'next_hop_num': 1,
        # prob settings
        'entropy_threshold': 1.0,
        'sample_num': 4,
        'sample_loss_weight': 0.001,
        # BERT settings
        'decay_factor': 0.9,
        'hidden_size': 64,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'intermediate_size': 128,
        'hidden_act': "gelu",
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'type_vocab_size': 1,
        'initializer_range': 0.1,
        'weight_decay': 0.01,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'max_seq_len': 5,
        'corr_epoch': 10,
        'corr_threshold': 1,
        'delete_target_ratio': 0.1,
        'delete_input_ratio': 0.1,
        'replace_ratio': 0.1,
        'rectify_input': True,
        'rectify_target': True,
        'replace_target': True,
        'self_ensemble': True,
        'temperature': 0.01,
        'weighted_loss': False,
        'rho': 0.99,
        # caser setting
        'n_h': 16,
        'n_v': 4,
        # GRU4Rec config
        'gru_layer_num': 2,
        'candidate_size': 2,
    }

random.seed(123)


def main():

    # for rec_model in ['BERT4Rec', 'FMLPRec', 'SASRec', 'GRU4Rec', 'Caser', 'FPMC']:
    for dataset in ['qk-vedio']:  # 'ml1m', 'beauty', 'yelp', 'qk-vedio'
        for rec_model in ['SASRec']: # ['BERT4Rec', 'FMLPRec', 'SASRec', 'GRU4Rec', 'Caser', 'FPMC']:
            config['dataset'] = dataset
            config['rec_model'] = rec_model
            data_model = GraphDataCollector(config)
            print_dict(config, 'config')
            trainer = Trainer(config, data_model, save_dir='./datasets/' + config['dataset'] + '/seq/')
            trainer.train()

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()




import torch


def print_dict(metrics, name):
    print('#' * 10 + ' ' + name + ' ' + '#' * 10)
    for key, value in metrics.items():
        print(f'{key}: {value}')


class RankingEvaluator(object):

    def __init__(self, test_batches):

        self.best_metrics = {
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
        self.best_iter = 0
        self.test_batches = test_batches

    def evaluate(self, model, train_iter, verbose=True, eval_model=0):
        model.eval()
        current_metrics = {
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
        # evaluate each batch
        test_size = 0
        for test_batch in self.test_batches:
            batch_metrics_values = model.eval_ranking(test_batch, eval_model)
            for metric, value in batch_metrics_values.items():
                current_metrics[metric] += value
            test_size += test_batch[0].shape[0]
        # summarize the metrics and update the best metrics
        for metric, value in current_metrics.items():
            avg_value = value / test_size
            current_metrics[metric] = avg_value
            if avg_value > self.best_metrics[metric]:
                self.best_metrics[metric] = avg_value
                self.best_iter = train_iter
        # print results
        if verbose:
            print_dict(current_metrics, 'current_metrics')
            print_dict(self.best_metrics, 'best_metrics')
        # early stop
        best_gap = train_iter - self.best_iter
        if best_gap > 40:
            if verbose:
                print('early stop')
            return False, current_metrics['ndcg_10']
        else:
            if verbose:
                print(f'no increase in recent {best_gap} iters')
            return True, current_metrics['ndcg_10']

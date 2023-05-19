import random
random.seed(0)
import time

import multiprocessing as mp
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from collections import Counter


def Most_Common(lst, topN=2):
    data = Counter(lst)
    list_of_item_with_frequency = data.most_common(topN)
    items = [list_of_item_with_frequency[i][0] for i in range(len(list_of_item_with_frequency))]
    return items

def list2str(lst):
    str_elements = [str(element) for element in lst]
    return ','.join(str_elements)


class SeqDataCollector(object):

    def __init__(self, config):
        print('#' * 10 + ' DataInfo ' + '#' * 10)
        self.config = config
        self.input_len = config['input_len']
        self.device = config['device']
        self.data_path = './datasets/' + config['dataset'] + '/seq/'
        random.seed(123)
        np.random.seed(123)
        self.user2Idx = {}
        self.item2Idx = {'mask': 0}
        self.itemIdx2Str = {}
        self.userIdx2sequence = {}
        self.item2succeeding_items = {}
        self.userItemSetTrain = {}
        self.itemUserList = {}
        self.item_freq = {}
        self.valid_users = set()
        self.valid_items = set()
        self.cpu_num = min(mp.cpu_count(), config['thread_num'])
        print(f'cpu num: {self.cpu_num}')

        self.numUser = 0
        self.numItem = 0
        self.item_item_content = None
        self.item_item_interaction = None
        # self.item_dist = [0 for _ in range(self.numItem)]
        # self.user_dist = [0 for _ in range(self.numUser)]

        build_succeeding = self.load_item_succeeding()
        self.load_seq_data(build_succeeding=build_succeeding)

        config['user_num'] = self.numUser
        config['item_num'] = self.numItem
        print(f'numUser:{self.numUser}')
        print(f'numItem:{self.numItem}')
        self.eval_neg_num = config['eval_neg_num']
        self.train_batch_size = config['train_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.train_size = 0
        self.valid_size = 0
        self.test_size = 0


    def load_seq_data(self, build_succeeding=True):
        full_seq_count = self.load_file('seq.dat')
        self.numUser = len(self.user2Idx)
        self.numItem = len(self.item2Idx)

        if build_succeeding:
            for user, item_seq in self.userIdx2sequence.items():
                train_item_seq = item_seq[0:-1]
                for i, item in enumerate(train_item_seq):
                    if item not in self.item2succeeding_items:
                        self.item2succeeding_items[item] = []
                    if i + 1 < len(train_item_seq) - 1:
                        succeeding_item = train_item_seq[i + 1]
                        self.item2succeeding_items[item].append(succeeding_item)
            for item, succeeding_items in self.item2succeeding_items.items():
                commonest_items = Most_Common(succeeding_items, self.config['candidate_size'])
                self.item2succeeding_items[item] = commonest_items
            self.save_item_succeeding(self.item2succeeding_items)

        for user, items in self.userIdx2sequence.items():
            self.userItemSetTrain[user] = set(items[0:-1])

    def load_item_succeeding(self):
        candidate_size = self.config['candidate_size']
        file_path = self.data_path + f'/item_succeeding{candidate_size}.dat'
        if os.path.exists(file_path):
            print(f'reading {file_path}')
            with open(file_path) as fin:
                for line in fin:
                    splited_line = line.strip().split(':')
                    str_succeeding_items = splited_line[1].strip().split(',')
                    item = int(splited_line[0])
                    succeeding_items = []
                    for str_item in str_succeeding_items:
                        if str_item == '':
                            continue
                        succeeding_items.append(int(str_item))
                    self.item2succeeding_items[item] = succeeding_items
            print(f'{file_path} loaded')
            return False
        else:
            return True

    def save_item_succeeding(self, item2succeedings):
        candidate_size = self.config['candidate_size']
        file_path = self.data_path + f'/item_succeeding{candidate_size}.dat'
        output_lines = []
        for item, succeeding_items in item2succeedings.items():
            output_line = f'{item}:' + list2str(succeeding_items) + '\n'
            output_lines.append(output_line)
        with open(file_path, 'w') as fout:
            fout.writelines(output_lines)
        print(f'{file_path} saved')

    def slide_window(self, itemList, window_size, candidate_size=2):
        """
        Input a sequence [1, 2, 3, 4, 5] with window size 3
        Return [0, 1, 2], [1, 2, 3],  [2, 3, 4],  [3, 4, 5]
        with   [0, 1, 1], [1, 1, 1],  [1, 1, 1],  [1, 1, 1]
        """
        new_item_list = [0] * (window_size - 2) + itemList
        num_seq = len(itemList) - 1
        for startIdx in range(num_seq):
            endIdx = startIdx + window_size
            item_sub_seq = new_item_list[startIdx:endIdx]

            mask = [1] * window_size
            for i, item in enumerate(item_sub_seq):
                if item == 0:
                    mask[i] = 0

            succeeding_items = []
            candidate_end_idx = min(endIdx + candidate_size, len(new_item_list))
            succeeding_items.extend(new_item_list[endIdx: candidate_end_idx])
            if len(succeeding_items) < candidate_size:
                second_last = item_sub_seq[-2]
                succeeding_items.extend(self.item2succeeding_items[second_last])
            if len(succeeding_items) < candidate_size:
                sample_num = candidate_size - len(succeeding_items)
                sampled_items = np.random.randint(low=1, high=self.numItem, size=sample_num)
                succeeding_items.extend(sampled_items)
            if len(succeeding_items) > candidate_size:
                succeeding_items = succeeding_items[0:candidate_size]

            assert len(succeeding_items) == candidate_size

            yield item_sub_seq, mask, succeeding_items[0: candidate_size]

    def generate_train_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating train samples')
        start = time.time()
        train_users = []
        train_hist_items = []
        train_masks = []
        train_targets = []
        train_succeedings = []
        sub_seq_len = input_len + 1
        abandon_count = 0

        instance_idx = 0
        target2instance_ids = [[] for _ in range(self.numItem)]
        for user, item_full_seq in self.userIdx2sequence.items():
            item_train_seq = item_full_seq[0:-1]
            for sub_seq, mask, succeeding_items in self.slide_window(item_train_seq, sub_seq_len):
                input_seq = sub_seq[0: input_len]
                input_mask = mask[0: input_len]
                target = sub_seq[input_len]
                assert len(sub_seq) == len(mask) == sub_seq_len
                # append lists
                train_users.append(user)
                train_hist_items.append(input_seq)
                train_masks.append(input_mask)
                train_targets.append(target)
                train_succeedings.append(succeeding_items)
                #
                target2instance_ids[target].append(instance_idx)
                instance_idx += 1

                self.valid_users.add(user)
                self.valid_items.add(target)
                for item in sub_seq:
                    if item != 0:
                        self.valid_items.add(item)
        self.train_size = len(train_users)
        assert self.train_size == instance_idx
        print(f"train_size: {self.train_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.train_size, 4)}) samples')
        print(f"valid user num: {len(self.valid_users)}")
        print(f"valid item num: {len(self.valid_items)}")

        dataset = UnidirectTrainDataset(self.config, train_users, train_hist_items,
                                        train_masks, train_targets, self.userItemSetTrain, target2instance_ids,
                                        self.valid_items, max_item_idx=self.numItem - 1,
                                        train_succeedings=train_succeedings)
        dataloader = DataLoader(dataset, shuffle=True, num_workers=self.cpu_num, batch_size=self.train_batch_size)

        return dataset, dataloader

    def generate_valid_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating valid samples')
        start = time.time()
        valid_users = []
        valid_hist_items = []
        valid_masks = []
        valid_targets = []
        abandon_count = 0
        for user, item_full_seq in self.userIdx2sequence.items():
            target_item = item_full_seq[-2]
            if user not in self.valid_users or target_item not in self.valid_items:
                abandon_count += 1
                continue
            valid_users.append(user)
            raw_input = item_full_seq[0:-2]
            raw_input_len = len(raw_input)
            if raw_input_len >= input_len:
                input = raw_input[-input_len:]
                mask = [1] * input_len
            else:  # raw_input_len < input_len
                input = [0] * (input_len - raw_input_len) + raw_input
                mask = [0] * (input_len - raw_input_len) + [1] * raw_input_len
            assert len(input) == len(mask) == input_len
            assert item_full_seq[-3] == input[-1]

            valid_hist_items.append(input)
            valid_masks.append(mask)
            valid_targets.append(target_item)

        dataset = UnidirectTrainDataset(valid_users, valid_hist_items,
                                        valid_masks, valid_targets, self.userItemSetTrain,
                                        max_item_idx=self.numItem - 1)
        dataloader = DataLoader(dataset, shuffle=True,
                                num_workers=self.cpu_num, batch_size=self.train_batch_size)
        self.valid_size = len(valid_users)
        print(f"valid_size: {self.valid_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.valid_size, 4)}) samples')
        return dataloader

    def generate_test_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating test samples')
        start = time.time()
        test_users = []
        test_hist_items = []
        test_masks = []
        test_targets = []
        pad_indices = []
        abandon_count = 0
        for user, item_full_seq in self.userIdx2sequence.items():
            test_users.append(user)
            raw_input = item_full_seq[0:-1]
            raw_input_len = len(raw_input)
            if raw_input_len >= input_len:
                input = raw_input[-input_len:]
                mask = [1] * input_len
            else: # raw_input_len < input_len
                input = [0] * (input_len - raw_input_len) + raw_input
                mask = [0] * (input_len - raw_input_len) + [1] * raw_input_len
            assert len(input) == len(mask) == input_len
            assert item_full_seq[-2] == input[-1]

            test_hist_items.append(input)
            test_masks.append(mask)

            sampled_negs = []

            if self.eval_neg_num == 'full':
                valid_neg_ids = self.valid_items - self.userItemSetTrain[user]
                pad_length = self.numItem - len(valid_neg_ids)
                valid_length = len(valid_neg_ids) + 1
                padded_valid_neg_ids = list(valid_neg_ids) + [0] * pad_length
                pad_indices.append([0] * valid_length + [-10^8] * pad_length)
                sampled_negs = padded_valid_neg_ids
            # else:
            #     num_item_to_rank = self.eval_neg_num + 1  # negative ones plus the positive one
            #     while len(sampled_negs) < num_item_to_rank:
            #         sampled_neg_cands = np.random.choice(self.numItem, self.eval_neg_num, False)
            #         valid_neg_ids = [x for x in sampled_neg_cands if x not in self.userItemSetTrain[user]]
            #         sampled_negs.extend(valid_neg_ids[:])
            #     sampled_negs = sampled_negs[:self.eval_neg_num]
            #     pad_indices.append([0] * num_item_to_rank)

            test_targets.append([item_full_seq[-1]] + list(sampled_negs))

        dataset = UnidirectTestDataset(test_users, test_hist_items, test_masks, test_targets, pad_indices)
        dataloader = DataLoader(dataset, shuffle=False,
                                num_workers=self.cpu_num, batch_size=self.test_batch_size)
        self.test_size = len(test_users)
        print(f"test_size: {self.test_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.test_size, 4)}) samples')
        return dataloader

    # def generate_item_dist(self):
    #     # print('generating item distribution')
    #     item_dist = np.array(self.item_dist)
    #     sum_click = item_dist.sum()
    #     self.item_dist = item_dist / sum_click
    #     # print(f'item dist 0: {self.item_dist[0]}')
    #     self.item_dist[0] = 0

    def load_file(self, file_name):
        file_path = self.data_path + '/' + file_name
        line_count = 0
        if os.path.exists(file_path):
            print(f'reading {file_name}')
            with open(file_path) as fin:
                for line in fin:
                    splited_line = line.strip().split(' ')
                    user, item = splited_line[0], splited_line[1]
                    if user not in self.user2Idx:
                        userIdx = len(self.user2Idx)
                        self.user2Idx[user] = userIdx
                    if item not in self.item2Idx:
                        itemIdx = len(self.item2Idx)
                        self.item2Idx[item] = itemIdx
                    userIdx = self.user2Idx[user]
                    itemIdx = self.item2Idx[item]
                    if userIdx not in self.userIdx2sequence:
                        self.userIdx2sequence[userIdx] = []
                    self.userIdx2sequence[userIdx].append(itemIdx)
                    line_count += 1
        return line_count

    def save_file(self, file_name):
        file_path = self.data_path + '/' + file_name
        line_count = 0
        if os.path.exists(file_path):
            print(f'reading {file_name}')
            with open(file_path) as fin:
                for line in fin:
                    splited_line = line.strip().split(' ')
                    user, item = splited_line[0], splited_line[1]
                    if user not in self.user2Idx:
                        userIdx = len(self.user2Idx)
                        self.user2Idx[user] = userIdx
                    if item not in self.item2Idx:
                        itemIdx = len(self.item2Idx)
                        self.item2Idx[item] = itemIdx
                    userIdx = self.user2Idx[user]
                    itemIdx = self.item2Idx[item]
                    if userIdx not in self.userIdx2sequence:
                        self.userIdx2sequence[userIdx] = []
                    self.userIdx2sequence[userIdx].append(itemIdx)
                    line_count += 1
        return line_count


class UnidirectTrainDataset(torch.utils.data.Dataset):

    def __init__(self, config, train_users, train_hist_items,
                 train_masks, train_targets, userItemSet, target2instance_ids, valid_items, max_item_idx, train_succeedings, aug_num=3):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        train_masks = [bs, seq_len]
        train_targets = [bs]
        clean_mask = [bs], denoting whether this instance is clean or not
        """
        assert len(train_users) == len(train_hist_items) == len(train_masks) == len(train_targets)
        self.config = config
        self.train_users = train_users
        self.train_hist_items = train_hist_items
        self.train_targets = train_targets
        # self.clean_mask = clean_mask
        self.train_size = len(train_users)
        self.userItemSet = userItemSet
        self.max_item_idx = max_item_idx
        # self.sample_neg_num = sample_neg_num + 1  # additional one for negative input sample
        # self.target_neg_num = sample_neg_num
        self.aug_num = aug_num
        self.input_len = len(self.train_hist_items[0])
        # self.item_dist = item_dist
        # self.numItem = len(item_dist)
        self.target2instance_ids = target2instance_ids
        self.train_succeedings = train_succeedings
        self.replacement_candidate_pool = [[] for i in range(len(train_users))]
        self.valid_items = valid_items
        self.init_replacement_candidate_pool()

    def init_replacement_candidate_pool(self):
        print('begin to initialize the replacement candidate pool')
        start = time.time()
        pool_size = 125
        for i, succeeding_items in enumerate(self.train_succeedings):
            sample_size = int(pool_size - len(succeeding_items))
            sampled_items = np.random.randint(low=1, high=self.max_item_idx, size=sample_size)
            self.replacement_candidate_pool[i] = np.append(succeeding_items, sampled_items)
        print(f'replacement candidate pool initialized successfully!, time: {time.time()-start}')

    def update_replacement_candidate_pool(self, indices, updated_candidates):
        for i in range(len(indices)):
            instance_idx = indices[i]
            new_candidates = updated_candidates[i, :]
            # sample_size = pool_size - len(old_candidates)
            # new_candidates = np.random.randint(low=1, high=self.max_item_idx, size=sample_size)
            self.replacement_candidate_pool[instance_idx] = new_candidates

    def __getitem__(self, index):
        userIdx = self.train_users[index]
        # interacted_items = self.userItemSet[userIdx]

        # final_negs = np.random.randint(low=1, high=self.numItem, size=self.sample_neg_num)

        # final_negs = []
        # while len(final_negs) < self.sample_neg_num:
        #     # [bs, num_neg]
        #     # sampled_negs = np.random.choice(self.numItem, self.neg_num, False, self.item_dist)
        #     sampled_negs = np.random.randint(low=1, high=self.numItem, size=self.sample_neg_num)
        #     valid_negs = [x for x in sampled_negs if x not in interacted_items]
        #     final_negs.extend(valid_negs[:])

        # final_negs_targets = final_negs[:self.target_neg_num]

        # target_for_neg_input = final_negs[-1]

        target = self.train_targets[index]
        # random_pos_instance_idx = random.choice(self.target2instance_ids[target])
        # random_neg_instance_idx = random.choice(self.target2instance_ids[target_for_neg_input])

        return userIdx, \
               torch.tensor(self.train_hist_items[index]), \
               target, \
               torch.tensor(self.replacement_candidate_pool[index]), \
               index



    def __len__(self):
        return self.train_size


class UnidirectTestDataset(torch.utils.data.Dataset):

    def __init__(self, test_users, test_hist_items, test_masks, test_targets, pad_indices=None):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        masks = [bs, seq_len]
        target_ids = [bs, pred_num]
        """
        self.test_users = test_users
        self.test_hist_items = test_hist_items
        self.test_targets = test_targets
        self.test_size = len(test_users)
        self.pad_indices = pad_indices

    def __getitem__(self, index):
        return self.test_users[index], \
               torch.tensor(self.test_hist_items[index]), \
               torch.tensor(self.test_targets[index]), \
               torch.tensor(self.pad_indices[index])

    def __len__(self):
        return self.test_size

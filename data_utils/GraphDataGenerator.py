from data_utils.SeqDataGenerator import *
import scipy.sparse as sp
from time import time

class GraphDataCollector(SeqDataCollector):

    def __init__(self, config):
        super().__init__(config)
        self.Graph = None
        self.hop_num = config['next_hop_num']

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.data_path + f'/s_pre_adj_mat_itemitem{self.hop_num}.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating UserItemNet and ItemItemNet")
                train_users, train_items = [], []

                # build ItemItemNet [I * I]
                ItemItemNet = None
                for hop in range(1, 1 + self.hop_num):
                    head_items, tail_items = [], []
                    for user, item_full_seq in self.userIdx2sequence.items():
                        item_train_seq = item_full_seq[0:-1]
                        seq_len = len(item_train_seq)
                        for i in range(seq_len):
                            if i + hop < seq_len:
                                head_items.append(item_train_seq[i])
                                tail_items.append(item_train_seq[i + hop])
                    if ItemItemNet is None:
                        ItemItemNet = sp.csr_matrix((np.ones(len(head_items)) * (1 / hop), (head_items, tail_items)),
                                  shape=(self.numItem, self.numItem))
                    else:
                        ItemItemNet += sp.csr_matrix((np.ones(len(head_items)) * (1 / hop), (head_items, tail_items)),
                                  shape=(self.numItem, self.numItem))
                print("generating adjacency matrix")
                s = time()
                adj_mat = ItemItemNet.tolil().todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"It costs {end - s}s to save norm_mat")
                sp.save_npz(self.data_path + f'/s_pre_adj_mat_itemitem{self.hop_num}.npz', norm_adj)

            self.Graph = self._convert_sp_mat_to_torch_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device).double()
            print("don't split the matrix")
        # DAD
        return self.Graph

    def _convert_sp_mat_to_torch_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
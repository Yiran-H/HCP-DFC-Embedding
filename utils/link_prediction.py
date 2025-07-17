import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import itertools
import math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.sparse import coo_matrix
from scipy import sparse
import numpy as np
import torch


def check_if_gpu():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = check_if_gpu()

L_list = [64]

def sample_zero_forever(mat):
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    while True:
        t = tuple(np.random.randint(0, mat.shape[0], 2))
        if t not in nonzero_or_sampled:
            yield t
            nonzero_or_sampled.add(t)


def sample_zero_n(mat, n=1000):
    itr = sample_zero_forever(mat)
    return [next(itr) for _ in range(n)]

def find_and_sample_zero_entries(sparse_matrix, num_samples=None):
    # Convert sparse matrix to dense format
    dense_matrix = sparse_matrix.toarray()

    # Create a boolean array: True where elements are zero
    zero_mask = dense_matrix == 0

    # Get the indices of zero elements
    zero_indices = np.argwhere(zero_mask)

    # Check if sampling is requested
    if num_samples is not None and num_samples > 0:
        # Ensure that there are enough zero entries to sample from
        if num_samples > len(zero_indices):
            raise ValueError("Requested more samples than available zeros.")
        # Sample indices randomly without replacement
        sampled_indices = zero_indices[np.random.choice(len(zero_indices), num_samples, replace=False)]
        return sampled_indices
    return zero_indices

def get_inf(data, mu_64, sigma_64, lookback,mult,train_indices):
    return_dict = {}
    #     for i in range (1, len(val_timestep) - 30):
    count = 0
    for ctr in train_indices[train_indices > lookback]:

# s0: 0, ... 110; 111,... 221; 222 ... 443
# s1: 444, ... 

# train_indices: 0,...221,444, ...

        A_node = data[ctr][0].shape[0]
        A = data[ctr][0]

        if count > 0:
            if A_node > A_prev_node:
                A = A[:A_prev_node, :A_prev_node]

            # if ctr < train_n and ctr > 0:
            if ctr > 0:

                ones_edj = A.nnz
                if A.shape[0] * mult <= (A.shape[0] - 1) * (A.shape[0] - 1):
                    zeroes_edj = A.shape[0] * mult
                else:
                    zeroes_edj = (A.shape[0] - 1) * (A.shape[0] - 1) - A.nnz

                tot = ones_edj + zeroes_edj

                # Ensure A is in COO format
                A_coo = A.tocoo() if not isinstance(A, coo_matrix) else A

                # Get the pairs directly from the COO format properties
                val_ones = list(zip(A_coo.row, A_coo.col))

                val_ones = list(map(list, val_ones))

                val_zeros = find_and_sample_zero_entries(A, zeroes_edj)

                val_edges = np.row_stack((val_ones, val_zeros))

                val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                a, b = unison_shuffled_copies(val_edges, val_ground_truth, count)

                if ctr >= 0:

                    a_embed = np.array(mu_64[ctr - (lookback + 1)])[a.astype(int)]

                    a_embed_stacked = np.vstack(a_embed)  # This stacks all [0] and [1] vertically

                    # Since we know every pair [0] and [1] are stacked sequentially, we can reshape:
                    n_features = a_embed.shape[2]  # Number of features in each sub-array
                    inp_clf_temp = a_embed_stacked.reshape(tot, 2 * n_features)

                    inp_clf = torch.tensor(inp_clf_temp)

                    inp_clf = inp_clf.to(device)
                    return_dict[ctr] = [inp_clf,b]
        A_prev_node = data[ctr][0].shape[0]
        count = count + 1
    return return_dict

def get_inf_a(data, mu_64, sigma_64, lookback,mult,train_indices):
    return_dict = {}
    #     for i in range (1, len(val_timestep) - 30):
    count = 0
    for ctr in train_indices[train_indices > lookback[0]]:

        A_node = data[ctr][0].shape[0]
        A = data[ctr][0]

        if count > 0:
            if A_node > A_prev_node:
                A = A[:A_prev_node, :A_prev_node]

            # if ctr < train_n and ctr > 0:
            if ctr > 0:

                ones_edj = A.nnz
                if A.shape[0] * mult <= (A.shape[0] - 1) * (A.shape[0] - 1):
                    zeroes_edj = A.shape[0] * mult
                else:
                    zeroes_edj = (A.shape[0] - 1) * (A.shape[0] - 1) - A.nnz

                tot = ones_edj + zeroes_edj

                # Ensure A is in COO format
                A_coo = A.tocoo() if not isinstance(A, coo_matrix) else A

                # Get the pairs directly from the COO format properties
                val_ones = list(zip(A_coo.row, A_coo.col))

                val_ones = list(map(list, val_ones))

                val_zeros = find_and_sample_zero_entries(A, zeroes_edj)

                val_edges = np.row_stack((val_ones, val_zeros))

                val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                a, b = unison_shuffled_copies(val_edges, val_ground_truth, count)

                if ctr >= 0:

                    a_embed = np.array(mu_64[ctr - lookback[ctr]])[a.astype(int)]

                    a_embed_stacked = np.vstack(a_embed)  # This stacks all [0] and [1] vertically

                    # Since we know every pair [0] and [1] are stacked sequentially, we can reshape:
                    n_features = a_embed.shape[2]  # Number of features in each sub-array
                    inp_clf_temp = a_embed_stacked.reshape(tot, 2 * n_features)

                    inp_clf = torch.tensor(inp_clf_temp)

                    inp_clf = inp_clf.to(device)
                    return_dict[ctr] = [inp_clf,b]
        A_prev_node = data[ctr][0].shape[0]
        count = count + 1
    return return_dict

def get_MAP_avg(mu_arr,sigma_arr,lookback,data,train_indices, test_indices):
    MAP_l = []
    MRR_l = []
    time_list = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

    for l_num in range(len(L_list)): # change to another name 

        mu_64 = mu_arr[l_num]
        sigma_64 = sigma_arr[l_num]


        class Classifier(torch.nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                activation = torch.nn.ReLU()

                self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=np.array(mu_64[0]).shape[1] * 2,
                                                               out_features=np.array(mu_64[0]).shape[1]),
                                               activation,
                                               torch.nn.Linear(in_features=np.array(mu_64[0]).shape[1],
                                                               out_features=1))

            def forward(self, x):
                return self.mlp(x)


        seed = 5
        torch.cuda.manual_seed_all(seed)
        classify = Classifier()
        classify.to(device)

        loss = torch.nn.BCEWithLogitsLoss(reduce=False)

        optim = torch.optim.Adam(classify.parameters(), lr=1e-3)
        mult = 10
        mult_test = 50
        num_epochs = 50
        return_dict = get_inf(data, mu_64, sigma_64, lookback,mult,train_indices)
        for epoch in range(num_epochs):
            #     for i in range (1, len(val_timestep) - 30):
            count = 0
            for ctr in train_indices[train_indices > lookback]:

                if count > 0:

                    # if ctr < train_n and ctr > 0:

                    if ctr >= 0:
                        classify.train()
                        decompose = return_dict[ctr]
                        inp_clf = decompose[0]
                        b = decompose[1]
                        out = classify(inp_clf).squeeze()

                        weight = torch.tensor([0.1, 0.9]).to(device)

                        label = torch.tensor(np.asarray(b)).to(device)

                        weight_ = weight[label.data.view(-1).long()].view_as(label)

                        l = loss(out, label)

                        l = l * weight_
                        l = l.mean()

                        optim.zero_grad()

                        l.backward()
                        optim.step()


                A_prev_node = data[ctr][0].shape[0]
                count = count + 1

        num_epochs = 1
        MAP_time = []
        MRR_time = []
        time_ctr = 0
        for epoch in range(num_epochs):
            get_MAP_avg = []
            get_MRR_avg = []

            #     for i in range (70, len(val_timestep)):
            count = 0

            for ctr in test_indices:

                A_node = data[ctr][0].shape[0]
                A = data[ctr][0]

                if count > 0:
                    if A_node > A_prev_node:
                        A = A[:A_prev_node, :A_prev_node]

                    if ctr >= 0:
                        # logging.debug('Testing')


                        ones_edj = A.nnz
                        if A.shape[0] * mult_test <= (A.shape[0] - 1) * (A.shape[0] - 1):
                            zeroes_edj = A.shape[0] * mult_test
                        else:
                            zeroes_edj = (A.shape[0] - 1) * (A.shape[0] - 1) - A.nnz

                        tot = ones_edj + zeroes_edj

                        val_ones = list(set(zip(*A.nonzero())))
                        val_ones = random.sample(val_ones, ones_edj)
                        val_ones = [list(ele) for ele in val_ones]
                        val_zeros = sample_zero_n(A, zeroes_edj)
                        val_zeros = [list(ele) for ele in val_zeros]
                        val_edges = np.row_stack((val_ones, val_zeros))

                        val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                        a, b = unison_shuffled_copies(val_edges, val_ground_truth, count)

                        if ctr > 0:

                            a_embed = np.array(mu_64[ctr - (lookback + 1)])[a.astype(int)]
                            a_embed_sig = np.array(sigma_64[ctr - (lookback + 1)])[a.astype(int)]

                            classify.eval()

                            inp_clf = []
                            for d_id in range(tot):
                                inp_clf.append(np.concatenate((a_embed[d_id][0], a_embed[d_id][1]), axis=0))

                            inp_clf = torch.tensor(np.asarray(inp_clf))

                            inp_clf = inp_clf.to(device)
                            with torch.no_grad():
                                out = classify(inp_clf).squeeze()

                            weight = torch.tensor([0.1, 0.9]).to(device)
                            #                         pos_weight = torch.ones([1])*9  # All weights are equal to 1

                            label = torch.tensor(np.asarray(b)).to(device)

                            weight_ = weight[label.data.view(-1).long()].view_as(label)

                            l = loss(out, label)

                            l = l * weight_
                            l = l.mean()

                            MAP_val = get_MAP_e(out.cpu(), label.cpu(), None)
                            get_MAP_avg.append(MAP_val)

                            MRR = get_MRR(out.cpu(), label.cpu(), np.transpose(a))

                            get_MRR_avg.append(MRR)

                            try:
                                if ctr == time_list[time_ctr]:
                                    MAP_time.append(MAP_val)
                                    MRR_time.append(MRR)
                                    time_ctr = time_ctr + 1
                            except:
                                pass

                            # logging.debug(
                            #     'Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}, Running Mean MAP: {}, Running Mean MRR: {}'.format(
                            #         epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None), MRR,
                            #         np.asarray(get_MAP_avg).mean(), np.asarray(get_MRR_avg).mean()))

                A_prev_node = data[ctr][0].shape[0]
                count = count + 1
        MAP_l.append(MAP_time)
        MRR_l.append(MRR_time)
        return np.asarray(get_MAP_avg).mean() , np.asarray(get_MRR_avg).mean()

def get_MAP_avg_a(mu_arr,sigma_arr,lookback,data,train_indices, test_indices):
    MAP_l = []
    MRR_l = []
    time_list = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

    for l_num in range(len(L_list)):

        mu_64 = mu_arr[l_num]
        sigma_64 = sigma_arr[l_num]


        class Classifier(torch.nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                activation = torch.nn.ReLU()

                self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=np.array(mu_64[0]).shape[1] * 2,
                                                               out_features=np.array(mu_64[0]).shape[1]),
                                               activation,
                                               torch.nn.Linear(in_features=np.array(mu_64[0]).shape[1],
                                                               out_features=1))

            def forward(self, x):
                return self.mlp(x)


        seed = 5
        torch.cuda.manual_seed_all(seed)
        classify = Classifier()
        classify.to(device)

        loss = torch.nn.BCEWithLogitsLoss(reduce=False)

        optim = torch.optim.Adam(classify.parameters(), lr=1e-3)
        mult = 10
        mult_test = 50
        num_epochs = 50
        return_dict = get_inf_a(data, mu_64, sigma_64, lookback,mult,train_indices)
        for epoch in range(num_epochs):
            #     for i in range (1, len(val_timestep) - 30):
            count = 0
            for ctr in train_indices[train_indices > lookback[0]]:

                if count > 0:

                    # if ctr < train_n and ctr > 0:

                    if ctr >= 0:
                        classify.train()
                        decompose = return_dict[ctr]
                        inp_clf = decompose[0]
                        b = decompose[1]
                        out = classify(inp_clf).squeeze()

                        weight = torch.tensor([0.1, 0.9]).to(device)

                        label = torch.tensor(np.asarray(b)).to(device)

                        weight_ = weight[label.data.view(-1).long()].view_as(label)

                        l = loss(out, label)

                        l = l * weight_
                        l = l.mean()

                        optim.zero_grad()

                        l.backward()
                        optim.step()


                A_prev_node = data[ctr][0].shape[0]
                count = count + 1

        num_epochs = 1
        MAP_time = []
        MRR_time = []
        time_ctr = 0
        for epoch in range(num_epochs):
            get_MAP_avg = []
            get_MRR_avg = []

            #     for i in range (70, len(val_timestep)):
            count = 0

            for ctr in test_indices:

                A_node = data[ctr][0].shape[0]
                A = data[ctr][0]

                if count > 0:
                    if A_node > A_prev_node:
                        A = A[:A_prev_node, :A_prev_node]

                    if ctr >= 0:
                        # logging.debug('Testing')


                        ones_edj = A.nnz
                        if A.shape[0] * mult_test <= (A.shape[0] - 1) * (A.shape[0] - 1):
                            zeroes_edj = A.shape[0] * mult_test
                        else:
                            zeroes_edj = (A.shape[0] - 1) * (A.shape[0] - 1) - A.nnz

                        tot = ones_edj + zeroes_edj

                        val_ones = list(set(zip(*A.nonzero())))
                        val_ones = random.sample(val_ones, ones_edj)
                        val_ones = [list(ele) for ele in val_ones]
                        val_zeros = sample_zero_n(A, zeroes_edj)
                        val_zeros = [list(ele) for ele in val_zeros]
                        val_edges = np.row_stack((val_ones, val_zeros))

                        val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

                        a, b = unison_shuffled_copies(val_edges, val_ground_truth, count)

                        if ctr > 0:

                            a_embed = np.array(mu_64[ctr - (lookback[ctr] + 1)])[a.astype(int)]
                            a_embed_sig = np.array(sigma_64[ctr - (lookback[ctr] + 1)])[a.astype(int)]

                            classify.eval()

                            inp_clf = []
                            for d_id in range(tot):
                                inp_clf.append(np.concatenate((a_embed[d_id][0], a_embed[d_id][1]), axis=0))

                            inp_clf = torch.tensor(np.asarray(inp_clf))

                            inp_clf = inp_clf.to(device)
                            with torch.no_grad():
                                out = classify(inp_clf).squeeze()

                            weight = torch.tensor([0.1, 0.9]).to(device)
                            #                         pos_weight = torch.ones([1])*9  # All weights are equal to 1

                            label = torch.tensor(np.asarray(b)).to(device)

                            weight_ = weight[label.data.view(-1).long()].view_as(label)

                            l = loss(out, label)

                            l = l * weight_
                            l = l.mean()

                            MAP_val = get_MAP_e(out.cpu(), label.cpu(), None)
                            get_MAP_avg.append(MAP_val)

                            MRR = get_MRR(out.cpu(), label.cpu(), np.transpose(a))

                            get_MRR_avg.append(MRR)

                            try:
                                if ctr == time_list[time_ctr]:
                                    MAP_time.append(MAP_val)
                                    MRR_time.append(MRR)
                                    time_ctr = time_ctr + 1
                            except:
                                pass

                            # logging.debug(
                            #     'Epoch: {}, Timestep: {}, Loss: {}, MAP: {}, MRR: {}, Running Mean MAP: {}, Running Mean MRR: {}'.format(
                            #         epoch, ctr, l.item(), get_MAP_e(out.cpu(), label.cpu(), None), MRR,
                            #         np.asarray(get_MAP_avg).mean(), np.asarray(get_MRR_avg).mean()))

                A_prev_node = data[ctr][0].shape[0]
                count = count + 1
        MAP_l.append(MAP_time)
        MRR_l.append(MRR_time)
        return np.asarray(get_MAP_avg).mean() , np.asarray(get_MRR_avg).mean()


def unison_shuffled_copies(a, b, seed):
    assert len(a) == len(b)
    np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_MAP_e(predictions,true_classes, adj):

    probs = predictions
    probs = probs.detach().cpu().numpy()
    true_classes = true_classes
    adj = adj

    var = average_precision_score(true_classes, probs)

    return var

def get_MRR(predictions,true_classes, adj):
    probs = predictions

    probs = probs.detach().cpu().numpy()
    true_classes = true_classes
    adj = adj


    pred_matrix = sp.coo_matrix((probs,(adj[0],adj[1]))).toarray()
    true_matrix = sp.coo_matrix((true_classes,(adj[0],adj[1]))).toarray()

    row_MRRs = []
    for i, pred_row in enumerate(pred_matrix):
            #check if there are any existing edges
        if np.isin(1,true_matrix[i]):
            row_MRRs.append(get_row_MRR(pred_row,true_matrix[i]))

    avg_MRR = torch.tensor(row_MRRs).mean()
    return avg_MRR

def get_row_MRR(probs,true_classes):
    existing_mask = true_classes == 1
        #descending in probability
    ordered_indices = np.flip(probs.argsort())

    ordered_existing_mask = existing_mask[ordered_indices]

    existing_ranks = np.arange(1,
                                   true_classes.shape[0]+1,
                                   dtype=np.float64)[ordered_existing_mask]

    MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
    return MRR


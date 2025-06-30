import numpy as np
import pandas as pd
import networkx as nx
import itertools
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import json
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from sklearn.model_selection import GroupShuffleSplit
import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        return json.load(f)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config()

class RMDataset_adaptive(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)


    def temp_process(self, data, lookback):
        dataset = {}

        num_graph_per_session = (1200 - config["window_size"]) // config["stride"] + 1
        for i in range(config["max_lb"], len(data)):
            session_start = (i // num_graph_per_session) * num_graph_per_session
            if i < session_start + config["max_lb"]:
                adj_matr = data[i][0].todense()
                B = np.zeros((config["dim_in"], lookback[i] + 1, config["dim_in"]))
                B[:adj_matr.shape[0], -1, :adj_matr.shape[1]] = adj_matr  # put it in the last frame
                dataset[i] = torch.tensor(B).clone().detach().requires_grad_(True).to(device)
            else:
                B = np.zeros((config["dim_in"], lookback[i] + 1, config["dim_in"]))
                for j in range(lookback[i] + 1):
                    adj_matr = data[i - lookback[i] + j][0].todense()
                    B[:adj_matr.shape[0], j, :adj_matr.shape[1]] = adj_matr
                dataset[i] = torch.tensor(B).clone().detach().requires_grad_(True).to(device)
        # Construct dict of hops and scale terms
        hop_dict = {}
        scale_terms_dict = {}
        print("Constructing dict of hops and scale terms")
        for i in range(config["max_lb"], len(data)):
            hops = get_hops(data[i][0], 2)
            scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                               hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
            hop_dict[i] = hops
            scale_terms_dict[i] = scale_terms

        # Construct dict of triplets
        triplet_dict = {}
        scale_dict = {}
        print("Constructing dict of triplets")
        for i in range(config["max_lb"], len(data)):
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            triplet_dict[i] = triplet
            scale_dict[i] = scale
        return dataset, triplet_dict, scale_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = torch.tensor(data, dtype=torch.float32)

        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]
        return x, triplet, scale
    
class RMDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)


    def temp_process(self, data, lookback):
        dataset = {}
        num_graph_per_session = (1200 - config["window_size"]) // config["stride"] + 1
        for i in range(lookback, len(data)):
            session_start = (i // num_graph_per_session) * num_graph_per_session
            if i < session_start + lookback:
                adj_matr = data[i][0].todense()
                B = np.zeros((config["dim_in"], lookback + 1, config["dim_in"]))
                B[:adj_matr.shape[0], -1, :adj_matr.shape[1]] = adj_matr  # put it in the last frame
                dataset[i] = torch.tensor(B).clone().detach().requires_grad_(True).to(device)
            else:
                B = np.zeros((config["dim_in"], lookback + 1, config["dim_in"]))
                for j in range(lookback + 1):
                    adj_matr = data[i - lookback + j][0].todense()
                    B[:adj_matr.shape[0], j, :adj_matr.shape[1]] = adj_matr
                dataset[i] = torch.tensor(B).clone().detach().requires_grad_(True).to(device)

        # Construct dict of hops and scale terms
        hop_dict = {}
        scale_terms_dict = {}
        print("Constructing dict of hops and scale terms")
        for i in range(lookback, len(data)):
            hops = get_hops(data[i][0], 2)
            scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                               hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                           for h in hops}
            hop_dict[i] = hops
            scale_terms_dict[i] = scale_terms

        # Construct dict of triplets
        triplet_dict = {}
        scale_dict = {}
        print("Constructing dict of triplets")
        for i in range(lookback, len(data)):
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            triplet_dict[i] = triplet
            scale_dict[i] = scale
        return dataset, triplet_dict, scale_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = torch.tensor(data, dtype=torch.float32)

        triplet = self.triplet_dict_data[idx]
        scale = self.scale_dict_data[idx]
        return x, triplet, scale

def get_graph_data(x):
    x = torch.tensor(x, dtype=torch.float32)

    # Create placeholders for edge_index and edge_attr
    edge_index_list = []
    edge_attr_list = []


    adj_matrix = x
    edge_index, edge_attr = dense_to_sparse(adj_matrix)
    edge_index_list.append(edge_index)
    edge_attr_list.append(edge_attr)

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)

    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, edge_index, edge_attr, batch

# Load config


def binary_tensor_to_data_format(binary_data):
    """
    binary_data: numpy array of shape [T, N, N], values in {0, 1}
    Returns: list of (A, X_Sparse) for each time step
    """
    time_len, num_nodes, _ = binary_data.shape
    data_list = []

    for t in range(time_len):
        adj_mat = binary_data[t]
        # diagonal set to 0
        np.fill_diagonal(adj_mat, 0)
    
        A = csr_matrix(adj_mat)
        
        # add self connection
        X = A + sp.eye(num_nodes)
        
        # set to torch sparse 
        X_sparse = sparse_feeder(X)
        X_sparse = spy_sparse2torch_sparse(X_sparse)

        data_list.append((A, X_sparse))
    
    return data_list

def sparse_feeder(M):
    
    M = sp.coo_matrix(M)
    return M


def spy_sparse2torch_sparse(data):
    """

    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples=data.shape[0]
    features=data.shape[1]
    values=data.data
    coo_data=data.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return t

# Prepare data function
def prepare_data_adaptive(dynfc_matrices, adaptive_lookbacks):
    dataset = {}
    hop_dict = {}
    scale_terms_dict = {}
    triplet_dict = {}
    scale_dict = {}

    num_windows, num_nodes, _ = dynfc_matrices.shape

    for i in range(len(adaptive_lookbacks)):
        lookback = adaptive_lookbacks[i]
        if i < lookback:
            continue

        B = np.zeros((num_nodes, lookback + 1, num_nodes))
        for j in range(lookback + 1):
            B[:, j, :] = dynfc_matrices[i - lookback + j]
        dataset[i] = B

        hops = get_hops(dynfc_matrices[i], lookback) # lookback = 2
        scale_terms = {
            h if h != -1 else max(hops.keys()) + 1:
            hops[h].sum(1) if h != -1 else hops[1].shape[0] - hops[h].sum(1)
            for h in hops
        }

        hop_dict[i] = hops
        scale_terms_dict[i] = scale_terms

        triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
        triplet_dict[i] = triplet
        scale_dict[i] = scale

    return dataset, hop_dict, scale_terms_dict, triplet_dict, scale_dict

def family_group(num_graph_per_subject):

    twins_df = pd.read_csv(config["twins_path"], header=None)
    twins_matrix = twins_df.values  # numpy array

    related_pairs = []
    unrelated_pairs = []
    identity_pairs = []

    n_subjects = twins_matrix.shape[0]

    for i in range(n_subjects):
        for j in range(i+1, n_subjects):  
            relation = twins_matrix[i, j]
            if relation == -1:
                identity_pairs.append((i, j))
            elif relation == 0:
                unrelated_pairs.append((i, j))
            elif relation > 1:
                related_pairs.append((i, j, relation))
    related_subjects = set()
    for i, j, _ in related_pairs:
        related_subjects.add(i)
        related_subjects.add(j)

    all_subjects = set(range(n_subjects))
    unrelated_subjects = list(all_subjects - related_subjects)

    # Group subjects by family
    G = nx.Graph()
    G.add_nodes_from(range(n_subjects))

    for i, j, _ in related_pairs:
        G.add_edge(i, j)

    family_groups = list(nx.connected_components(G))

    subject_to_group = {}
    for group_idx, group in enumerate(family_groups):
        for subj in group:
            subject_to_group[subj] = group_idx

    for subj in unrelated_subjects:
        subject_to_group[subj] = len(family_groups) + subj 

    max_index = max(max(group) for group in family_groups)
    groups = np.full(max_index + 1, -1, dtype=int)
    for group_id, group in enumerate(family_groups):
        for idx in group:
            groups[idx] = group_id

    filtered_groups = []
    for group in family_groups:
        valid_members = group.intersection(set(range(config["num_subjects"])))
        if valid_members:  
            filtered_groups.append(valid_members)

    # shuffle + split
    np.random.seed(42)
    np.random.shuffle(filtered_groups)

    n_total = len(filtered_groups)
    n_train = int(n_total * config["train_ratio"])
    n_val   = int(n_total * config["val_ratio"])
    n_test  = n_total - n_train - n_val

    train_groups = filtered_groups[:n_train]
    val_groups   = filtered_groups[n_train:n_train+n_val]
    test_groups  = filtered_groups[n_train+n_val:]

    # sample index（471 per subject）
    def subjects_to_sample_indices(subject_set):
        sample_indices = []
        for subj in subject_set:
            sample_indices.extend(range(subj * num_graph_per_subject, (subj + 1) * num_graph_per_subject))
        return sample_indices

    train_subjects = set.union(*train_groups)
    val_subjects   = set.union(*val_groups)
    test_subjects  = set.union(*test_groups)
    # print(train_subjects)
    # print(val_subjects)
    # print(test_subjects)

    train_indices = np.array(subjects_to_sample_indices(train_subjects))
    val_indices   = np.array(subjects_to_sample_indices(val_subjects))
    test_indices  = np.array(subjects_to_sample_indices(test_subjects))
    
    return train_indices, val_indices, test_indices

def split_by_ratio_per_sub(graphs_per_subject=444):

    train_indices = []
    val_indices = []
    test_indices = []

    for subj_id in range(config["num_subjects"]):
        start = subj_id * graphs_per_subject
        end = start + graphs_per_subject

        train_end = start + int(config["train_ratio"] * graphs_per_subject)
        val_end = train_end + int(config["val_ratio"] * graphs_per_subject)

        train_indices.extend(range(start, train_end))
        val_indices.extend(range(train_end, val_end))
        test_indices.extend(range(val_end, end))

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def split_by_session(graphs_per_session=111):
    train_indices = []
    val_indices = []
    test_indices = []

    for subj_id in range(config["num_subjects"]):
        base = subj_id * 4 * graphs_per_session
        for s in range(4):
            session_start = base + s * graphs_per_session
            session_end = session_start + graphs_per_session
            if s in [0, 1]: 
                train_indices.extend(range(session_start, session_end))
            elif s == 2: 
                val_indices.extend(range(session_start, session_end))
            elif s == 3: 
                test_indices.extend(range(session_start, session_end))
    
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def prepare_data(dynfc_matrices, lookback):
    dataset = {}
    hop_dict = {}
    scale_terms_dict = {}
    triplet_dict = {}
    scale_dict = {}

    num_windows, num_nodes, _ = dynfc_matrices.shape

    for i in range(lookback, num_windows):
        B = np.zeros((num_nodes, lookback + 1, num_nodes))
        for j in range(lookback + 1):
            adj_matr = dynfc_matrices[i - lookback + j]
            B[:, j, :] = adj_matr
        dataset[i] = B

    for i in range(lookback, num_windows):
        hops = get_hops(dynfc_matrices[i], lookback) #!!!!
        scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                       hops[h].sum(1) if h != -1 else hops[1].shape[0] - hops[h].sum(1)
                       for h in hops}
        hop_dict[i] = hops
        scale_terms_dict[i] = scale_terms

        triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
        triplet_dict[i] = triplet
        scale_dict[i] = scale

    return dataset, hop_dict, scale_terms_dict, triplet_dict, scale_dict

# # Define function to sample last hop
# def sample_last_hop(dynfc_matrices, nodes):
#     num_windows, num_nodes, _ = dynfc_matrices.shape
#     sampled = np.random.randint(0, num_nodes, len(nodes))
#     for window_idx in range(num_windows):
#         window_matrix = dynfc_matrices[window_idx]
#         nnz = np.nonzero(window_matrix[nodes, sampled])[0]
#         while len(nnz) != 0:
#             new_sample = np.random.randint(0, num_nodes, len(nnz))
#             sampled[nnz] = new_sample
#             nnz = np.nonzero(window_matrix[nnz, new_sample])[0]
#     return sampled

# # Define function to sample all hops
# def sample_all_hops(hops, nodes=None):
#     N = hops[1].shape[0]
#     if nodes is None:
#         nodes = np.arange(N)
#     sampled_nodes = []
#     for node in nodes:
#         node_samples = [node]
#         for h in hops.keys():
#             if h != -1:
#                 if len(hops[h]) == 0:
#                     node_samples.append(-1)
#                 else:
#                     node_samples.append(np.random.choice(hops[h][node]))
#         node_samples.append(np.random.randint(0, N))
#         sampled_nodes.append(node_samples)
#     return np.array(sampled_nodes)

# # Define function to get hops
# def get_hops(dynfc_matrices, K=2):
#     num_windows, num_nodes = dynfc_matrices.shape
#     hops = {1: dynfc_matrices.copy(), -1: dynfc_matrices.copy()}
#     np.fill_diagonal(hops[1], 0)
#     for h in range(2, K + 1):
#         next_hop = np.dot(hops[h - 1], dynfc_matrices)
#         next_hop[next_hop > 0] = 1
#         for prev_h in range(1, h):
#             next_hop -= np.multiply(next_hop, hops[prev_h])
#         np.fill_diagonal(next_hop, 0)
#         hops[h] = next_hop
#         hops[-1] += next_hop
#     return hops

# # Define function to convert to triplets
# def to_triplets(sampled_nodes, scale_terms):
#     triplets = []
#     triplet_scale_terms = []
#     for i, j in itertools.combinations(np.arange(1, sampled_nodes.shape[1]), 2):
#         triplet = sampled_nodes[:, [0] + [i, j]]
#         triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
#         triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
#         triplets.append(triplet)
#         triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])
#     return np.row_stack(triplets), np.concatenate(triplet_scale_terms)

def get_hops(A, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hopness to consider.

    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    """
    hops = {1: A.tolil(), -1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop
        hops[-1] += next_hop 

    return hops


def sample_last_hop(A, nodes):
    """
    For each node in nodes samples a single node from their last (K-th) neighborhood.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix encoding which nodes belong to any of the 1, 2, ..., K-1, neighborhoods of every node
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N]
        The sampled nodes.
    """
    N = A.shape[0]

    sampled = np.random.randint(0, N, len(nodes))

    nnz = A[nodes, sampled].nonzero()[1]
    while len(nnz) != 0:
        new_sample = np.random.randint(0, N, len(nnz))
        sampled[nnz] = new_sample
        #nnz = A[nodes,sampled].nonzero()[1]
        nnz = A[nnz, new_sample].nonzero()[1]

    return sampled


def sample_all_hops(hops, nodes=None):
    """
    For each node in nodes samples a single node from all of their neighborhoods.

    Parameters
    ----------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N, K]
        The sampled nodes.
    """

    N = hops[1].shape[0]

    if nodes is None:
        nodes = np.arange(N)

    return np.vstack((nodes,
                      np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
                                for h in hops.keys() if h != -1]),
                      sample_last_hop(hops[-1], nodes)
                      )).T


def to_triplets(sampled_hops, scale_terms):
    """
    Form all valid triplets (pairwise constraints) from a set of sampled nodes in triplets

    Parameters
    ----------
    sampled_hops : array-like, shape [N, K]
       The sampled nodes.
    scale_terms : dict
        The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood

    Returns
    -------
    triplets : array-like, shape [?, 3]
       The transformed triplets.
    """
    triplets = []
    triplet_scale_terms = []

    for i, j in itertools.combinations(np.arange(1, sampled_hops.shape[1]), 2):
        triplet = sampled_hops[:, [0] + [i, j]]
        triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
        triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
        triplets.append(triplet)

        triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])

    return np.row_stack(triplets), np.concatenate(triplet_scale_terms)


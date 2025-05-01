import numpy as np
import itertools


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

        hops = get_hops(dynfc_matrices[i], lookback)
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

# Define function to sample last hop
def sample_last_hop(dynfc_matrices, nodes):
    num_windows, num_nodes, _ = dynfc_matrices.shape
    sampled = np.random.randint(0, num_nodes, len(nodes))
    for window_idx in range(num_windows):
        window_matrix = dynfc_matrices[window_idx]
        nnz = np.nonzero(window_matrix[nodes, sampled])[0]
        while len(nnz) != 0:
            new_sample = np.random.randint(0, num_nodes, len(nnz))
            sampled[nnz] = new_sample
            nnz = np.nonzero(window_matrix[nnz, new_sample])[0]
    return sampled

# Define function to sample all hops
def sample_all_hops(hops, nodes=None):
    N = hops[1].shape[0]
    if nodes is None:
        nodes = np.arange(N)
    sampled_nodes = []
    for node in nodes:
        node_samples = [node]
        for h in hops.keys():
            if h != -1:
                if len(hops[h]) == 0:
                    node_samples.append(-1)
                else:
                    node_samples.append(np.random.choice(hops[h][node]))
        node_samples.append(np.random.randint(0, N))
        sampled_nodes.append(node_samples)
    return np.array(sampled_nodes)

# Define function to get hops
def get_hops(dynfc_matrices, K=2):
    num_windows, num_nodes = dynfc_matrices.shape
    hops = {1: dynfc_matrices.copy(), -1: dynfc_matrices.copy()}
    np.fill_diagonal(hops[1], 0)
    for h in range(2, K + 1):
        next_hop = np.dot(hops[h - 1], dynfc_matrices)
        next_hop[next_hop > 0] = 1
        for prev_h in range(1, h):
            next_hop -= np.multiply(next_hop, hops[prev_h])
        np.fill_diagonal(next_hop, 0)
        hops[h] = next_hop
        hops[-1] += next_hop
    return hops

# Define function to convert to triplets
def to_triplets(sampled_nodes, scale_terms):
    triplets = []
    triplet_scale_terms = []
    for i, j in itertools.combinations(np.arange(1, sampled_nodes.shape[1]), 2):
        triplet = sampled_nodes[:, [0] + [i, j]]
        triplet = triplet[(triplet[:, 1] != -1) & (triplet[:, 2] != -1)]
        triplet = triplet[(triplet[:, 0] != triplet[:, 1]) & (triplet[:, 0] != triplet[:, 2])]
        triplets.append(triplet)
        triplet_scale_terms.append(scale_terms[i][triplet[:, 1]] * scale_terms[j][triplet[:, 2]])
    return np.row_stack(triplets), np.concatenate(triplet_scale_terms)

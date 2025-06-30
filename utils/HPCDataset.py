import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, random_split
import json
from utils.prepare_data import *

config = load_config()

import numpy as np
import h5py
import torch
import networkx as nx
from torch.utils.data import Dataset, random_split

class HPCDataset(Dataset):
    def __init__(self, mat_file_path, fc_key="FCmats", ts_key="tc_clean", threshold=0.4, window_size=100, step_size=10, selected_sessions=[0], subject_num=1):
        super().__init__()
        self.mat_data = h5py.File(mat_file_path, "r")
        
        tc_clean = self.mat_data[ts_key]
        num_sessions, num_subjects = tc_clean.shape
        ts_data = [[np.array(self.mat_data[ref]) for ref in tc_clean[i]] for i in range(num_sessions)]
        ts_data = np.transpose(ts_data, (1, 0, 2, 3))  # [subjects, sessions, regions, timepoints]
        self.ts_data = np.array(ts_data)

        # ================== Compute Temporal Graphs ================== #
        self.temporal_weighted_graphs = []  # weighted FC matrices
        self.temporal_binary_graphs = []    # binarized FC matrices

        _, FC_mats_list = self.compute_temporal_graphs(
            self.ts_data, 
            subject_num=subject_num, 
            selected_sessions=selected_sessions, 
            window_size=window_size, 
            step_size=step_size
        ) # [subjects, regions, regions, total_graphs]
        # binary_mats = self.compute_binary_temporal_graphs(FC_mats, threshold)

        binary_mats = []
        for FC_mats in FC_mats_list:
            binary_matrices = self.compute_binary_temporal_graphs(FC_mats, threshold)
            binary_mats.append(binary_matrices)
        binary_mats = np.concatenate(binary_mats, axis=0)


        # Rearrange FC_mats to (timepoints, regions, regions)
        weighted_graph = np.transpose(FC_mats, (2, 0, 1))  # (num_graphs, regions, regions)

        self.temporal_weighted_graphs.append(weighted_graph)
        self.temporal_binary_graphs.append(binary_mats)

    def __len__(self):
        return len(self.ts_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ts_data[idx], dtype=torch.float32),
            torch.tensor(self.temporal_weighted_graphs[idx], dtype=torch.float32),
            torch.tensor(self.temporal_binary_graphs[idx], dtype=torch.float32)
        )
    def compute_temporal_graphs(self, ts_data, subject_num, selected_sessions=[0], window_size=100, step_size=10):
        all_subject_graphs = []
        all_subject_FC_mats = []

        for subject_idx in range(subject_num):
            subject_data = ts_data[subject_idx]  # [sessions, regions, timepoints]

            subject_graphs = []
            subject_FC_list = []

            for s in selected_sessions:
                session_data = subject_data[s]  # shape: [regions, timepoints]
                total_timepoints = session_data.shape[1]

                # Normalize session independently
                session_data = (session_data - np.mean(session_data, axis=1, keepdims=True)) / \
                            np.std(session_data, axis=1, keepdims=True)

                num_graphs = (total_timepoints - window_size) // step_size + 1
                FC_mats = np.zeros((session_data.shape[0], session_data.shape[0], num_graphs))

                graphs = []
                graph_idx = 0
                for start in range(0, total_timepoints - window_size + 1, step_size):
                    window = session_data[:, start:start + window_size]
                    corr_matrix = np.corrcoef(window, rowvar=True)
                    np.fill_diagonal(corr_matrix, 0)

                    FC_mats[:, :, graph_idx] = corr_matrix
                    G = nx.from_numpy_array(corr_matrix)
                    graphs.append(G)

                    graph_idx += 1

                subject_graphs.extend(graphs)
                subject_FC_list.append(FC_mats)

            # Concatenate FC matrices across sessions
            subject_FC_mats = np.concatenate(subject_FC_list, axis=2)  # shape: [regions, regions, total_graphs]
            all_subject_graphs.append(subject_graphs)
            all_subject_FC_mats.append(subject_FC_mats)

        return all_subject_graphs, all_subject_FC_mats


    def compute_binary_temporal_graphs(self, FC_mats, threshold=0.1):
        binary_matrices = []
        for i in range(FC_mats.shape[2]):  # iterate over time points
            corr_matrix = FC_mats[:, :, i]
            binary_matrix = (corr_matrix > threshold).astype(int)
            binary_matrices.append(binary_matrix)
        return np.array(binary_matrices)


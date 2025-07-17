import torch_geometric.transforms as T
import os

# from utils import *
import pickle
import json
# from exp_mod import get_MAP_avg
import ray
from ray import tune
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

from utils.brain_info import *

from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, precision_recall_curve

import warnings
warnings.filterwarnings("ignore")

import itertools

from torch.nn import ELU,Dropout

from mamba_ssm import Mamba, Mamba2
from tqdm import tqdm


from torch.nn.utils import clip_grad_norm_

from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

from torch_geometric.nn import GINEConv, GINConv,  global_add_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

# from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

torch.backends.cudnn.deterministic=True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df_atlas = load_atlas()
meta = built_node_meta(map_roi_id_subnetwork(load_roi_dict(df_atlas)))
print(meta)
print(device)


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

def get_graph_data_id(x):
    """
    x: numpy array or torch tensor of shape (N, N), adjacency matrix
    node_meta: torch tensor of shape (N, 2), where each row is [hemisphere, subnetwork] for that node
    """
    edge_index_list = []
    x = torch.tensor(x, dtype=torch.float32)
    node_meta = torch.tensor(meta, dtype=torch.long)  # shape: [N, 2]

    # Convert dense adjacency to edge_index and edge_attr
    edge_index, edge_weight = dense_to_sparse(x)  # edge_index: [2, E], edge_weight: [E]
    edge_index_list.append(edge_index)
    edge_index = torch.cat(edge_index_list, dim=1)

    # For each edge, look up source and target node attributes
    source_nodes_roi = torch.tensor(index_to_roi_id(edge_index[0], df_atlas), dtype=torch.long)
    target_nodes_roi = torch.tensor(index_to_roi_id(edge_index[1], df_atlas), dtype=torch.long)

    # Get hemisphere and subnet info
    source_meta = node_meta[source_nodes_roi - 1]  # shape: [E, 2]
    target_meta = node_meta[target_nodes_roi - 1]  # shape: [E, 2]

    # Combine: [source_hemi, source_subnet, target_hemi, target_subnet]
    edge_attr = torch.cat([source_meta[:, 0:1],  # source hemisphere
                           source_meta[:, 1:2],  # source subnet
                           target_meta[:, 0:1],  # target hemisphere
                           target_meta[:, 1:2]], dim=1)  # shape: [E, 4]

    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x.to(device), edge_index.to(device), edge_attr.to(device), batch.to(device)


# def val_loss(t):
#     l = []
#     for j in range(63, 72):
#         _, muval, sigmaval = t(val_data[j])
#         val_l = build_loss(triplet_dict[j], scale_dict[j], muval, sigmaval, 64, scale=False)
#         l.append(val_l.cpu().detach().numpy())
#     return np.mean(l)


def Energy_KL(mu, sigma, pairs, L):
    ij_mu = mu[pairs]
    ij_sigma = sigma[pairs]
    sigma_ratio = ij_sigma[:, 1] / (ij_sigma[:, 0] + 1e-14)
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), 1)
    mu_diff_sq = torch.sum(torch.square(ij_mu[:, 0] - ij_mu[:, 1]) / (ij_sigma[:, 0] + 1e-14), 1)
    return 0.5 * (trace_fac + mu_diff_sq - L - log_det)


# Define loss function
def build_loss(triplets, scale_terms, mu, sigma, L, scale):
    hop_pos = torch.stack([torch.tensor(triplets[:, 0]), torch.tensor(triplets[:, 1])], 1).type(torch.int64)
    hop_neg = torch.stack([torch.tensor(triplets[:, 0]), torch.tensor(triplets[:, 2])], 1).type(torch.int64)
    eng_pos = Energy_KL(mu, sigma, hop_pos, L)
    eng_neg = Energy_KL(mu, sigma, hop_neg, L)
    energy = torch.square(eng_pos) + torch.exp(-eng_neg)
    if scale:
        loss = torch.mean(energy * torch.Tensor(scale_terms).cpu())
    else:
        loss = torch.mean(energy)
    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class ApplyConv(torch.nn.Module):
    def __init__(self,mamba_attn,dropout, channels: int, conv: Optional[MessagePassing], norm: Optional[str] = 'batch_norm', norm_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.conv = conv
        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.dropout = dropout # add this line to set dropout rate
        self.mamba = mamba_attn
        self.pe_lin = Linear(2, channels)

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            edge_attr: Optional[Tensor] = None,
            **kwargs,
    ) -> Tensor:

        hs = []
        if self.conv is not None:
            h = self.conv(x, edge_index, edge_attr, **kwargs)
            # h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x #96,96
            h = self.norm1(h)
            hs.append(h)

        # inp_mamba = x.reshape(1,x.size(0), x.size(1)) #1,96,96  Batch , time stamp , features

        # h = self.mamba(inp_mamba) # 
        # h = h.mean(dim=0) #96,96
        # hs.append(h)

        out = sum(hs) #96,96
        return out

class MambaG2G1(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G1, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        self.mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])

        self.conv_mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])
        # Define a sequential model for GINEConv
        nn_model = Sequential(Linear(dim_in, dim_in), ReLU(), Linear(dim_in, dim_in))

        # Correctly instantiate GINEConv with the sequential model
        self.conv = ApplyConv(self.conv_mamba,dropout,dim_in, GINEConv(nn_model))
        # self.conv = ApplyConv(self.conv_mamba,dropout,dim_in, GINConv(nn_model))
        # self.enc_input_fc = nn.Linear(dim_in, dim_in) 
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)
        self.edge_emb = Embedding(2, dim_in)

    def forward(self, input):
        # e = self.enc_input_fc(input)
        z = []
        for i in range(input.size(1)):
            x = input[:, i, :]
            x, edge_index, edge_attr, batch = get_graph_data(x)
            # print(i,": ", edge_attr)
            edge_attr = self.edge_emb(edge_attr.int()) # label hem/subnetwork
            x = self.conv(x,edge_index, batch=batch, edge_attr=edge_attr)
            z.append(x)
        z = torch.stack(z, 1)
        e = self.mamba(z)
        e = e.mean(dim=1)  # Average pooling to maintain the expected shape
        e = self.dropout(e)  # Apply dropout after average pooling
        x = torch.tanh(self.out_fc(e))
        x = self.elu(x)
        x = self.dropout(x)  # Apply dropout after the activation
        # x = x[:, :92]
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        sigma = self.elu(sigma) + 1 + 1e-14

        return x, mu, sigma
    

class MambaG2G2(torch.nn.Module):
    def __init__(self, config, dim_in, dim_out, dropout=0.2):
        super(MambaG2G2, self).__init__()
        self.D = dim_in
        self.elu = nn.ELU()
        # self.mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])
        # self.conv_mamba = Mamba(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'])

        self.mamba = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=config['d_model'], # Model dimension d_model
            d_state=config['d_state'],  # SSM state expansion factor, typically 64 or 128
            d_conv=config['d_conv'],    # Local convolution width
            headdim=config['headdim'],
            d_ssm=None,
            expand=2,       # This is a multiplier
            ngroups=6       # 96*2 / 6 = 32 (divisible by 8)
        )
        self.conv_mamba = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=config['d_model'], # Model dimension d_model
            d_state=config['d_state'],  # SSM state expansion factor, typically 64 or 128
            d_conv=config['d_conv'],    # Local convolution width
            headdim=config['headdim'],
            d_ssm=None,
            expand=2,       # This is a multiplier
            ngroups=6       # 96*2 / 6 = 32 (divisible by 8)
        )

        # Define a sequential model for GINEConv
        nn_model = Sequential(Linear(dim_in, dim_in), ReLU(), Linear(dim_in, dim_in))

        # Correctly instantiate GINEConv with the sequential model
        self.conv = ApplyConv(self.conv_mamba,dropout,dim_in, GINEConv(nn_model))
        # self.conv = ApplyConv(self.conv_mamba,dropout,dim_in, GINConv(nn_model))

        # self.enc_input_fc = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(p=dropout)  # Add Dropout layer
        self.out_fc = nn.Linear(config['d_model'], self.D)  # Adjusted to match output dimension
        self.sigma_fc = nn.Linear(self.D, dim_out)
        self.mu_fc = nn.Linear(self.D, dim_out)
        self.edge_emb = Embedding(2, dim_in)
        self.hem_emb = nn.Embedding(2, 24)
        self.subnet_emb = nn.Embedding(17, 24)

    def forward(self, input):
        z = []
        for i in range(input.size(1)):
            x = input[:, i, :]
            # x, edge_index, edge_attr, batch = get_graph_data(x)
            # edge_attr = self.edge_emb(edge_attr.int())

            x, edge_index, edge_attr, batch = get_graph_data_id(x) # edge_attr shape: [E, 4]
            a1 = self.hem_emb(edge_attr[:, 0])
            a2 = self.subnet_emb(edge_attr[:, 1])
            b1 = self.hem_emb(edge_attr[:, 2])
            b2 = self.subnet_emb(edge_attr[:, 3])
            edge_attr = torch.concatenate([a1, a2, b1, b2], dim=1)
            # print(edge_attr.shape)
            x = self.conv(x,edge_index, batch=batch, edge_attr=edge_attr)
            z.append(x)
        # print("z shape 1:", z.shape)
        z = torch.stack(z, 1)  
        # print("z shape 2:", z.shape)
        e = self.mamba(z) # if lb is fixed: L N D; L_b N D; posi embedding+ self-attention + fixed lb
        # print("e shape after mamba:", e.shape)
        e = e.mean(dim=1)  # Average pooling to maintain the expected shape
        e = self.dropout(e)  # Apply dropout after average pooling
        x = torch.tanh(self.out_fc(e))
        x = self.elu(x)
        x = self.dropout(x)  # Apply dropout after the activation
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        sigma = self.elu(sigma) + 1 + 1e-14

        return x, mu, sigma

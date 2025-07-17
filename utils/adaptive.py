import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import torch
from utils.prepare_data import load_config


config = load_config()

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)   
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_adaptive(binary_graph, num_graph_per_sub):
    print("computing adaptive lookback:")
    adaptive_lookbacks = []
    for subject_idx in range(config["num_subjects"]):
        start = subject_idx * num_graph_per_sub
        end = start + num_graph_per_sub
        subject_graphs = binary_graph[start:end]  # shape: (444, 92, 92)

        lookbacks = []
        session_size = num_graph_per_sub // 4
        for i in range(4): 
            session_graphs = subject_graphs[i * session_size: (i + 1) * session_size]
            novelty_indices, _ = compute_novelty_index(session_graphs)  # list of len 111
            session_lookbacks = [config["max_lb"]] + quantile_binning(novelty_indices[1:])
            lookbacks.extend(session_lookbacks)
            
        adaptive_lookbacks.extend(lookbacks)
    adaptive_lookbacks = np.array(adaptive_lookbacks)
    print(adaptive_lookbacks[:100])
    print("adaptive lookback computed over!")
    return adaptive_lookbacks

# ========================= Novelty Index Computation ========================= #

def compute_novelty_index(dynfc_matrices):
    """
    Compute novelty index over a sequence of dynamic functional connectivity (dFC) matrices.

    Parameters:
    dynfc_matrices : list or ndarray
        List of dynamic functional connectivity matrices (binary or weighted).

    Returns:
    novelty_indices : list
        List of novelty indices per time step.
    average_novelty_index : float
        Average novelty index across all time steps.
    """
    novelty_indices = []
    seen_edges = set()

    for matrix in dynfc_matrices:
        current_graph = nx.from_numpy_array(matrix)
        current_edges = set(current_graph.edges())

        new_edges = current_edges - seen_edges
        total_edges = len(current_edges)

        if total_edges > 0:
            novelty_index = len(new_edges) / total_edges
        else:
            novelty_index = 0

        novelty_indices.append(novelty_index)
        seen_edges.update(current_edges)

    average_novelty_index = np.mean(novelty_indices)
    return novelty_indices, average_novelty_index

# ========================= Quantile Binning for Adaptive Lookback ========================= #

def quantile_binning(novelties, min_lookback=1, max_lookback=config["max_lb"]):
    """
    Map novelty indices to adaptive lookback values using quantile binning.

    Parameters:
    novelties : list or ndarray
        List of novelty indices.
    min_lookback : int
        Minimum lookback value.
    max_lookback : int
        Maximum lookback value.

    Returns:
    lookbacks : list
        List of adaptive lookback values corresponding to each novelty index.
    """
    novelties = np.array(novelties)
    num_bins = max_lookback - min_lookback + 1
    quantiles = np.quantile(novelties, np.linspace(0, 1, num_bins + 1))

    lookbacks = []
    for n in novelties:
        for i in range(num_bins):
            if quantiles[i] <= n <= quantiles[i + 1]:
                if config["adaptive_type"]:
                    lookbacks.append(min_lookback + i)
                else:
                    lookbacks.append(max_lookback - i)
                break
    return lookbacks

# ========================= Plot Lookback vs Novelty ========================= #

def plot_adaptive_lookback_vs_novelty(novelty_indices, adaptive_lookbacks):
    """
    Plot adaptive lookback values against true novelty indices.

    Parameters:
    novelty_indices : list
        List of novelty indices.
    adaptive_lookbacks : list
        List of adaptive lookback values.
    """
    fig, ax1 = plt.subplots(figsize=(16, 5))

    # Left y-axis: Lookback
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("Lookback", fontsize=12, color="#3498DB")
    ax1.plot(adaptive_lookbacks, label="Adaptive Lookback (Quantile Binning)", color="#3498DB", marker="o", linewidth=2)
    ax1.tick_params(axis='y', labelcolor="#3498DB")
    ax1.set_ylim(0, 5)

    # Right y-axis: true novelty values scaled
    ax2 = ax1.twinx()
    ax2.set_ylabel("Novelty Index", fontsize=12, color="#E74C3C")
    ax2.plot(np.array(novelty_indices) * 5, label="Novelty Index", color="#E74C3C", linewidth=2)
    ax2.set_ylim(0, 5)
    ax2_ticks = np.linspace(0, 5, 6)
    ax2.set_yticks(ax2_ticks)
    ax2.set_yticklabels([f"{tick / 5:.1f}" for tick in ax2_ticks])
    ax2.tick_params(axis='y', labelcolor="#E74C3C")

    # Title and grid
    plt.title("Adaptive Lookback vs. True Novelty Index", fontsize=14, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.show()


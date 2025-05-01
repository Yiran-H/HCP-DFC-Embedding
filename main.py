import numpy as np
import torch
import json
from utils.prepare_data import *
from models.GDGMamba import *
from utils.HPCDataset import *

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve, auc
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import time

# Load config
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

# Load dataset and prepare FC matrix
def load_dataset(config):
    dataset = HPCDataset(
        mat_file_path=config["data"]["dataset_path"]["hcp1003"],
        fc_key="FCmats",
        ts_key="tc_clean",
        threshold=0.4,
        window_size=100,
        step_size=10,
        selected_sessions=[0]
    )
    _, _, binary_graph = dataset[0]
    return np.array(binary_graph)

# Prepare training, validation, test data
def split_data(dataset, lookback, binary_graph, device):
    train_size = int(0.7 * len(binary_graph))
    val_size = int(0.2 * len(binary_graph))

    train_end = lookback + train_size
    val_start = train_end
    val_end = val_start + val_size
    test_start = val_end
    test_end = len(binary_graph)

    train_data = {i: torch.tensor(dataset[i], dtype=torch.float32).to(device) for i in range(lookback, train_end)}
    val_data = {i: torch.tensor(dataset[i], dtype=torch.float32).to(device) for i in range(val_start, val_end)}
    test_data = {i: torch.tensor(dataset[i], dtype=torch.float32).to(device) for i in range(test_start, test_end)}

    print(f"Training Data: {lookback} to {train_end}")
    print(f"Validation Data: {val_start} to {val_end}")
    print(f"Test Data: {test_start} to {test_end}")

    return train_data, val_data, test_data

# Training loop with early stopping and scheduler
def train_model(model, optimizer, scheduler, train_data, val_data, epochs, weight_decay, device, patience, hop_dict, scale_terms_dict):
    loss_mainlist, val_mainlist = [], []
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for e in range(epochs):
        loss_list = []
        model.train()
        for i in train_data.keys():
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            optimizer.zero_grad()
            inputs = train_data[i]
            _, mu, sigma = model(inputs)
            loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += weight_decay * l2_reg

            loss_list.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss_value = 0.0
        val_samples = 0
        for i in val_data.keys():
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            inputs = val_data[i]
            _, mu, sigma = model(inputs)
            val_loss_value += build_loss(triplet, scale, mu, sigma, 64, scale=False).item()
            val_samples += 1
        val_loss_value /= val_samples

        loss_mainlist.append(np.mean(loss_list))
        val_mainlist.append(val_loss_value)
        print(f"Epoch: {e}, Average Training Loss: {loss_mainlist[-1]}, Validation Loss: {val_mainlist[-1]}")

        scheduler.step(val_loss_value)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            early_stopping_counter = 0
            torch.save(model.state_dict(), config["saved_model_path"])
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered")
                break

    return loss_mainlist, val_mainlist

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(6, 4))
    plt.semilogy(train_losses, label='Training Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Semilogy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results/plot/loss_curve_semilogy.png", dpi=300)
    plt.show()

# Evaluate test loss
def evaluate_test_loss(model, test_data, hop_dict, scale_terms_dict):
    model.eval()
    test_loss_value = 0.0
    test_samples = 0
    with torch.no_grad():
        for i in test_data.keys():
            triplet, scale = to_triplets(sample_all_hops(hop_dict[i]), scale_terms_dict[i])
            inputs = test_data[i]
            _, mu, sigma = model(inputs)
            test_loss_value += build_loss(triplet, scale, mu, sigma, 64, scale=False).item()
            test_samples += 1
    return test_loss_value / test_samples

# Compute MAP and MRR
def compute_map_mrr(model, test_data, binary_graph, device):
    embeddings_list, labels_list = [], []
    for i in test_data.keys():
        inputs = test_data[i]
        with torch.no_grad():
            _, embeddings, _ = model(inputs.to(device))
        embeddings_flat = embeddings.cpu().numpy().reshape(-1, embeddings.shape[-1])
        embeddings_list.append(embeddings_flat)
        labels_list.append(np.array(binary_graph)[i].astype(int).flatten())

    X = np.concatenate(embeddings_list, axis=0)
    y = np.concatenate(labels_list, axis=0)[:X.shape[0]]

    X, y = RandomOverSampler(random_state=42).fit_resample(X, y)
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=123)
    clf.fit(X, y)

    AP_list, RR_list = [], []
    for i in test_data.keys():
        inputs = test_data[i]
        with torch.no_grad():
            _, embeddings, _ = model(inputs.to(device))
        X_test = embeddings.cpu().numpy().reshape(-1, embeddings.shape[-1])
        y_score = clf.predict_proba(X_test)[:, 1]
        y_true = np.array(binary_graph)[i].astype(int).flatten()[:len(y_score)]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        AP_list.append(auc(recall, precision))

        true_indices = np.where(y_true == 1)[0]
        if len(true_indices) == 0:
            RR_list.append(0.0)
        else:
            sorted_indices = np.argsort(y_score)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                if idx in true_indices:
                    RR_list.append(1.0 / rank)
                    break
            else:
                RR_list.append(0.0)

    MAP = np.mean(AP_list)
    MRR = np.mean(RR_list)
    return MAP, MRR


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    binary_graph = load_dataset(config)
    dataset, hop_dict, scale_terms_dict, _, _ = prepare_data(binary_graph, config["lookback"])

    train_data, val_data, test_data = split_data(dataset, config["lookback"], binary_graph, device)

    model = MambaG2G1(config["GDGMamba1"], config["dim_in"], config["dim_out"], dropout=config["dropout"]).to(device)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8)

    start = time.time()
    train_losses, val_losses = train_model(model, optimizer, scheduler, train_data, val_data,
                                           config["epochs"],
                                           config["weight_decay"], device, config["patience"],
                                           hop_dict, scale_terms_dict)
    print("Time taken:", time.time() - start)

    plot_loss_curves(train_losses, val_losses)

    # Test loss evaluation
    model.load_state_dict(torch.load(config["saved_model_path"]))
    model.eval()
    test_loss = evaluate_test_loss(model, test_data, hop_dict, scale_terms_dict)
    print(f"Test Loss: {test_loss:.4f}")

    # MAP & MRR evaluation
    MAPs, MRRs = [], []
    for run in range(5):
        print(f"Run {run+1}...")
        MAP, MRR = compute_map_mrr(model, test_data, binary_graph, device)
        MAPs.append(MAP)
        MRRs.append(MRR)

    print(f"\n==== Final Results over 5 runs ====")
    print(f"MAP: Mean = {np.mean(MAPs):.4f}, Variance = {np.var(MAPs):.6f}")
    print(f"MRR: Mean = {np.mean(MRRs):.4f}, Variance = {np.var(MRRs):.6f}")

if __name__ == "__main__":
    main()

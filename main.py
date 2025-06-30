import numpy as np
import torch
from utils.prepare_data import *
from models.GDGMamba import *
from utils.HPCDataset import *
from utils.link_prediction import *

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve, auc
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import time

    
# Load dataset and prepare FC matrix
def load_dataset(config):
    dataset = HPCDataset(
        mat_file_path=config["data"]["dataset_path"]["hcp1003"],
        fc_key="FCmats",
        ts_key="tc_clean",
        threshold=config["threshold"],
        window_size=config["window_size"],
        step_size=config["stride"],
        selected_sessions=[0,1,2,3],
        subject_num=config["num_subjects"]
    )
    _, _, binary_graph = dataset[0]
    return np.array(binary_graph)

def optimise_mamba(data, config):

    # model = MambaG2G1(config["GDGMamba1"], config["dim_in"], config["dim_out"], dropout=config["dropout"]).to(device)
    model = MambaG2G2(config["GDGMamba2"], config["GDGMamba2"]["d_model"], config["dim_out"], dropout=config["dropout"]).to(device)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    dataset = RMDataset(data, config["lookback"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    loss_mainlist, val_mainlist = [], []
    best_val_loss = float('inf')
    early_stopping_counter = 0

    start = time.time()

    for e in tqdm(range(50)):
        model.train()
        loss_step = []
        for i in train_indices[train_indices >= lookback]:
            x, triplet, scale = dataset[i]
            optimizer.zero_grad()
            # x = x.clone().detach().requires_grad_(True).to(device)
            _,mu, sigma = model(x)
            loss = build_loss(triplet, scale, mu, sigma, config["dim_out"], scale=False)

            loss_step.append(loss.cpu().detach().numpy())
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss_value = 0.0
        val_samples = 0
        with torch.no_grad():
            for i in val_indices:
                x, triplet, scale = dataset[i]
                _, mu, sigma = model(x)
                val_loss_value += build_loss(triplet, scale, mu, sigma, config["dim_out"], scale=False).item()
                val_samples += 1
        val_loss_value /= val_samples

        loss_mainlist.append(np.mean(loss_step))
        val_mainlist.append(val_loss_value)
        print(f"Epoch: {e}, Average Training Loss: {loss_mainlist[-1]}, Validation Loss: {val_mainlist[-1]}")

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            early_stopping_counter = 0

            save_path = f"./models/lookback_{lookback}/dim_out_{config['dim_out']}/model_100_ratio.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(model.state_dict(), save_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config["patience"]:
                print("Early stopping triggered")
                break
    print("Training Time taken: ", time.time() - start)

    return model


config = load_config()
lookback = config["lookback"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if os.path.exists("binary_graph_100_ratio.npy"):
    binary_graph = np.load("binary_graph_100_ratio.npy")
else:
    binary_graph = load_dataset(config)
    np.save("binary_graph_100_ratio.npy", binary_graph)

num_graph_per_sub = binary_graph.shape[0] // config["num_subjects"]
print("binary_graph:", binary_graph.shape)
print("num_graph_per_sub:", num_graph_per_sub)
# train_indices, val_indices, test_indices = family_group(num_graph_per_sub)
train_indices, val_indices, test_indices = split_by_ratio_per_sub(num_graph_per_sub)

# for mamba2:
original_data = np.array(binary_graph)  # shape (111, 92, 92)
padded_data = np.zeros((len(binary_graph), 96, 96), dtype=original_data.dtype)
padded_data[:, :92, :92] = original_data
binary_graph = padded_data

data = binary_tensor_to_data_format(binary_graph)

model = optimise_mamba(data,config)
model = MambaG2G2(config["GDGMamba2"], config["GDGMamba2"]["d_model"], config["dim_out"], dropout=config["dropout"]).to(device)

dataset = RMDataset(data, lookback)
#read the best_model.pt
# model.load_state_dict(torch.load('best_model.pth'))

mu_timestamp = []
sigma_timestamp = []

save_path = f"./models/lookback_{lookback}/dim_out_{config['dim_out']}/model_100_ratio.pth"
model.load_state_dict(torch.load(save_path))
with torch.no_grad():
    model.eval()
    for i in range(lookback, binary_graph.shape[0]):
        x, triplet, scale = dataset[i]
        x = x.clone().detach().requires_grad_(True).to(device)
        _, mu, sigma = model(x)
        mu_timestamp.append(mu.cpu().detach().numpy())
        sigma_timestamp.append(sigma.cpu().detach().numpy())
name = 'Results/RealityMining'
save_sigma_mu = True
sigma_L_arr = []
mu_L_arr = []
if save_sigma_mu == True:
    sigma_L_arr.append(sigma_timestamp)
    mu_L_arr.append(mu_timestamp)

import time
start = time.time()
MAPS = []
MRR = []

for i in tqdm(range(1)):
    curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback,data, train_indices, test_indices)
    MAPS.append(curr_MAP)
    MRR.append(curr_MRR)
#print mean and std of map and mrr
print("Mean MAP: ", np.mean(MAPS))
print("Mean MRR: ", np.mean(MRR))
print("Std MAP: ", np.std(MAPS))
print("Std MRR: ", np.std(MRR))
print("Reference Time taken: ", (time.time() - start) / 5)




#
# def train_model(config):
#     map_value = optimise_mamba(data,lookback = config['lookback'], dim_in=config['dim_in'], d_conv=config['d_conv'],d_state=config['d_state'],dropout=config['dropout'],lr=config['lr'],weight_decay=config['weight_decay'],walk_length=walk)
#     return map_value
#
#
# def objective(config):  # ①
#     while True:
#         acc = train_model(config)
#         train.report({"MAP": acc})  # Report to Tunevvvvvvv
#
#
# ray.init(  runtime_env={
#             "working_dir": str(os.getcwd()),
#         })  # Initialize Ray
#
#
# search_space = {"lr": tune.loguniform(1e-5, 1e-2)
#         ,"dim_in": tune.randint(16, 100),
#         "lookback": tune.randint(1,5),
#                 "d_conv": tune.randint(2, 10),
#                 "d_state": tune.randint(2, 50),
#                 "dropout": tune.uniform(0.1, 0.5),
#                 "weight_decay": tune.loguniform(1e-5, 1e-3)}
#
# # Create an Optuna search space
# algo = OptunaSearch(
# )
#
#
# tuner = tune.Tuner(  # ③
#     tune.with_resources(
#         tune.with_parameters(objective),
#         resources={"gpu": 0.25}
#     ),
#     tune_config=tune.TuneConfig(
#         metric="MAP",
#         mode="max",
#         search_alg=algo,
#         num_samples=100
#     ),
#     param_space=search_space,
#     run_config=train.RunConfig(
#         stop={"training_iteration": 1}  # Limit the training iterations to 1
#     )
# )
#
# results = tuner.fit()
# print("Best config is:", results.get_best_result().config)

#{'lr': 2.6152417925517384e-05, 'dim_in': 71, 'lookback': 4, 'd_conv': 8, 'd_state': 6, 'dropout': 0.14669919601710057, 'weight_decay': 3.427957936022128e-05}


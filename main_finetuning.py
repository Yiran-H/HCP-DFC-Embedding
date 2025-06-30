import numpy as np
import torch
import json
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

class RMDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback
        # self.data_len = data.shape[0]
        self.dataset,self.triplet_dict_data, self.scale_dict_data = self.temp_process(data, lookback)


    def temp_process(self, data, lookback):
        dataset = {}
        for i in range(lookback, len(data)):
            B = np.zeros((96, lookback + 1, 96))
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
        selected_sessions=[0],
        subject_num=1
    )
    _, _, binary_graph = dataset[0]
    return np.array(binary_graph)

def optimise_mamba(data, config):

    # model = MambaG2G1(config["GDGMamba1"], config["dim_in"], config["dim_out"], dropout=config["dropout"]).to(device)
    model = MambaG2G2(config_o["GDGMamba2"], 96, 64, dropout=config["dropout"]).to(device)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    # Create dataset
    dataset = RMDataset(data, config["lookback"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    for e in tqdm(range(50)):
        model.train()
        loss_step = []
        for i in range(config["lookback"], train_num):
                x, triplet, scale = dataset[i]
                optimizer.zero_grad()
                # x = x.clone().detach().requires_grad_(True).to(device)
                _,mu, sigma = model(x)
                loss = build_loss(triplet, scale, mu, sigma, 64, scale=False)

                loss_step.append(loss.cpu().detach().numpy())
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
    f_MAP = []
    for i in range(3):
        mu_timestamp = []
        sigma_timestamp = []
        with torch.no_grad():
            model.eval()
            for i in range(config["lookback"], len(data)):
                x, triplet, scale = dataset[i]
                x = x.clone().detach().requires_grad_(False).to(device)
                _, mu, sigma = model(x)
                mu_timestamp.append(mu.cpu().detach().numpy())
                sigma_timestamp.append(sigma.cpu().detach().numpy())
    
        # Save mu and sigma matrices
        name = 'Results/RealityMining'
        save_sigma_mu = True
        sigma_L_arr = []
        mu_L_arr = []
        if save_sigma_mu == True:
            sigma_L_arr.append(sigma_timestamp)
            mu_L_arr.append(mu_timestamp)
    
        MAP,_ = get_MAP_avg(mu_L_arr, sigma_L_arr, config["lookback"],data, train_num, val_num, total_samples)
        f_MAP.append(MAP)

    return np.mean(f_MAP)


#{'lr': 2.2307858381535968e-05, 'dim_in': 49, 'lookback': 4, 'd_conv': 3, 'd_state': 6, 'dropout': 0.17661562119283333, 'weight_decay': 1.466563344626497e-05}
# lookback = 2
config_o = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

binary_graph = load_dataset(config_o)

np.random.seed(42)
indices = np.random.permutation(binary_graph.shape[0])
binary_graph = binary_graph[indices]


total_samples = binary_graph.shape[0]

train_num = int(total_samples * config["train_ratio"])
val_num = int(total_samples * config["val_ratio"])
test_num = total_samples - train_num - val_num  

# for mamba2:
original_data = np.array(binary_graph)  # shape (111, 92, 92)
padded_data = np.zeros((len(binary_graph), 96, 96), dtype=original_data.dtype)
padded_data[:, :92, :92] = original_data
binary_graph = padded_data
print(binary_graph.shape)
# # dataset, hop_dict, scale_terms_dict, _, _ = prepare_data(binary_graph, config["lookback"])
# # train_data, val_data, test_data = split_data(dataset, config["lookback"], binary_graph, device)

data = binary_tensor_to_data_format(binary_graph)

# model = optimise_mamba(data,config)

# dataset = RMDataset(data, config["lookback"])
# #read the best_model.pt
# # model.load_state_dict(torch.load('best_model.pth'))
# mu_timestamp = []
# sigma_timestamp = []
# with torch.no_grad():
#     model.eval()
#     for i in range(lookback, 111):
#         x, triplet, scale = dataset[i]
#         x = x.clone().detach().requires_grad_(True).to(device)
#         _, mu, sigma = model(x)
#         mu_timestamp.append(mu.cpu().detach().numpy())
#         sigma_timestamp.append(sigma.cpu().detach().numpy())
# name = 'Results/RealityMining'
# save_sigma_mu = True
# sigma_L_arr = []
# mu_L_arr = []
# if save_sigma_mu == True:
#     sigma_L_arr.append(sigma_timestamp)
#     mu_L_arr.append(mu_timestamp)

# import time
# start = time.time()
# MAPS = []
# MRR = []
# for i in tqdm(range(5)):
#     curr_MAP, curr_MRR = get_MAP_avg(mu_L_arr, sigma_L_arr, lookback,data)
#     MAPS.append(curr_MAP)
#     MRR.append(curr_MRR)
# #print mean and std of map and mrr
# print("Mean MAP: ", np.mean(MAPS))
# print("Mean MRR: ", np.mean(MRR))
# print("Std MAP: ", np.std(MAPS))
# print("Std MRR: ", np.std(MRR))
# print("Time taken: ", time.time() - start)




def train_model(config):
    map_value = optimise_mamba(data,config)
    return map_value


def objective(config):  # ①
    while True:
        acc = train_model(config)
        train.report({"MAP": acc})  # Report to Tunevvvvvvv


ray.init(  runtime_env={
            "working_dir": str(os.getcwd()),
            'excludes': ['/hcp1003_RestALL_Schaefer_tcCLEAN.mat']
        })  # Initialize Ray


search_space = {"lr": tune.loguniform(1e-7, 1e-3),
        # ,"dim_in": tune.randint(16, 100),
                "lookback": 3,
                # "d_conv": tune.randint(2, 10),
                # "d_state": tune.randint(2, 50),
                "dropout": tune.uniform(0.1, 0.5),
                "weight_decay": tune.loguniform(1e-6, 1e-3)}

# Create an Optuna search space
algo = OptunaSearch(
)


tuner = tune.Tuner(  # ③
    tune.with_resources(
        tune.with_parameters(objective),
        resources={"gpu": 0.25}
    ),
    tune_config=tune.TuneConfig(
        metric="MAP",
        mode="max",
        search_alg=algo,
        num_samples=30
    ),
    param_space=search_space,
    run_config=train.RunConfig(
        stop={"training_iteration": 1}  # Limit the training iterations to 1
    )
)

results = tuner.fit()
print("Best config is:", results.get_best_result().config)

# {'lr': 2.6152417925517384e-05, 'dim_in': 71, 'lookback': 4, 'd_conv': 8, 'd_state': 6, 'dropout': 0.14669919601710057, 'weight_decay': 3.427957936022128e-05}


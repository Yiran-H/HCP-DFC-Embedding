import numpy as np
import torch
import json
from utils.prepare_data import *
from models.GDGMamba import *
from utils.HPCDataset import *
# from utils.link_prediction import *
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


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

config = load_config()
lookback = config["lookback"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
subject_embeddings = []  # (20, embedding_dim)
subject_ages = []  # (20,)

behav_data = pd.read_csv(config["behavioral_path"], sep=',', header=None)
print('Bahavior data size:', behav_data.shape)
ages = behav_data.iloc[:, 0].values[:config["num_subjects"]]

# binary_graph = load_dataset(config)
binary_graph = np.load("binary_graph_100_ratio.npy")

# np.random.seed(42)
# indices = np.random.permutation(binary_graph.shape[0])
# binary_graph = binary_graph[indices]

# for mamba2:
original_data = np.array(binary_graph)  # shape (111, 92, 92)
padded_data = np.zeros((len(binary_graph), 96, 96), dtype=original_data.dtype)
padded_data[:, :92, :92] = original_data
binary_graph = padded_data

# dataset, hop_dict, scale_terms_dict, _, _ = prepare_data(binary_graph, config["lookback"])
# train_data, val_data, test_data = split_data(dataset, config["lookback"], binary_graph, device)

data = binary_tensor_to_data_format(binary_graph)
dataset = RMDataset(data, config["lookback"])

model = MambaG2G2(config["GDGMamba2"], config["GDGMamba2"]["d_model"], config["dim_out"], dropout=config["dropout"]).to(device)
save_path = f"./models/lookback_{lookback}/dim_out_{config['dim_out']}/model_100_ratio.pth"
model.load_state_dict(torch.load(save_path))

with torch.no_grad():
    model.eval()
    for subject_idx in range(config["num_subjects"]):
        embeddings_per_subject = []
        
        for t in range(443): 
            i = subject_idx * 443 + t + lookback
            x, triplet, scale = dataset[i]
            x = x.clone().detach().to(device)
            _, mu, sigma = model(x)  # mu: [96, 64]

            embeddings_per_subject.append(mu.cpu().numpy())  # shape: (96, 64)

        embeddings_per_subject = np.stack(embeddings_per_subject, axis=0)  # (471, 96, 64)
        subject_embeddings.append(embeddings_per_subject)
        subject_ages.append(ages[subject_idx])

subject_embeddings = np.stack(subject_embeddings, axis=0)  # (100, 470, 96, 64)

np.save("subject_embeddings.npy", subject_embeddings)
np.save("subject_ages.npy", subject_ages)

'''
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)



class ConvReducer(nn.Module):
    def __init__(self, in_dim=config["dim_out"], mid_channels=2048, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, mid_channels, kernel_size=3, padding=1),  # Input: (B, 64, 471)
            nn.ReLU(),
            nn.Conv1d(mid_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → (B, mid_channels, 1)
        )
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):  # x: (B, T, ROI, F) = (100, 471, 96, 64)
        x = x.mean(dim=2)              # mean over ROI → (B, T, F) = (100, 471, 64)
        x = x.permute(0, 2, 1)         # → (B, F, T) = (100, 64, 471)
        x = self.conv(x)               # → (B, mid_channels, 1)
        x = x.squeeze(-1)              # → (B, mid_channels)
        return self.fc(x)              # → (B, out_dim)


subject_embeddings = np.load("subject_embeddings.npy")
subject_ages = np.load("subject_ages.npy")  # (100,)

print("Subject embedding shape:", subject_embeddings.shape)
# print("Subject ages shape:", subject_ages.shape)


tensor_input = torch.tensor(subject_embeddings, dtype=torch.float32)
model_conv = ConvReducer(in_dim=config["dim_out"], out_dim=128)

reduced = []
batch_size = 10
for i in range(0, tensor_input.shape[0], batch_size):
    batch = tensor_input[i:i+batch_size]
    with torch.no_grad():
        reduced_batch = model_conv(batch)
    reduced.append(reduced_batch.cpu())

reduced_output = torch.cat(reduced).numpy()


# # Step 2: Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     reduced_output, subject_ages, test_size=0.2, random_state=42
# )

# Step 3: Define regression models
n_cores = -1  # all cores
regression_models = { # fine tune / multiple runs
    'ElasticNet': ElasticNet(alpha=1e-05, l1_ratio=0.9, max_iter=10000, random_state=42),
    'Ridge': Ridge(alpha=0.000215, random_state=42),
    'Lasso': Lasso(alpha=0.0001, max_iter=10000, random_state=42), 
    'RandomForest': RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1,
        max_features='sqrt',
        min_samples_split=5
    ),
    'SVR': SVR(C=10, epsilon=0.1, kernel='rbf', gamma='scale')
}

n_runs = 5
for name, reg in regression_models.items():
    maes = []
    mses = []
    r2s = []
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            reduced_output, subject_ages, test_size=0.2, random_state=42 + run
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        #print("run ", run + 1, ": ", mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
        maes.append(mean_absolute_error(y_test, y_pred))
        mses.append(mean_squared_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))
    
    print(f"\nModel: {name}")
    print(f"MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
    print(f"MSE: {np.mean(mses):.2f} ± {np.std(mses):.2f}")
    print(f"R^2: {np.mean(r2s):.2f} ± {np.std(r2s):.2f}")

# Step 3: Define models with hyperparameter grids

'''
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

model_grids = {
    'ElasticNet': {
        'model': ElasticNet(max_iter=10000, random_state=42),
        'params': {
            'alpha': np.logspace(-5, -2, 4),       # e.g. [1e-5, 1e-4, 1e-3, 1e-2]
            'l1_ratio': [0.01, 0.1, 0.5, 0.9]
        }
    },
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {
            'alpha': np.logspace(-5, 3, 7)         # [1e-5, 1e-3, 1e-1, 1e1, 1e3]
        }
    },
    'Lasso': {
        'model': Lasso(max_iter=10000, random_state=42),
        'params': {
            'alpha': np.logspace(-6, 0, 7)         # [1e-6, 1e-4, 1e-2, 1, 10]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 0.3]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 0.1],
            'kernel': ['rbf'],
            'epsilon': [0.05, 0.1]
        }
    }
}

X_train, X_test, y_train, y_test = train_test_split(
    reduced_output, subject_ages, test_size=0.2
)

# Step 4: Grid search and evaluation
for name, item in model_grids.items():
    print(f"Tuning {name}...")
    grid = GridSearchCV(item['model'], item['params'], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"MAE on test: {mae:.2f}")
    print(f"MSE on test: {mse:.2f}")     
    print(f"R² on test: {r2:.2f}")
'''

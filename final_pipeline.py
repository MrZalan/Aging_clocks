import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import FeatureAgglomeration
from sklearn.base import BaseEstimator, RegressorMixin

# Linear & Classical Models
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV, TweedieRegressor, BayesianRidge, HuberRegressor, ARDRegression
from sklearn.svm import LinearSVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Trees & Ensembles
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal

# Deep Learning (PyTorch & TF)
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from pytorch_tabnet.tab_model import TabNetRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

import shap


# -- Experiment Control --
DATASET = 'temporal_pole'         # Options: 'bhak', 'temporal_pole', 'orb_cortex'
FEATURE_EXTRACTOR = 'selectkbest' # Options: 'selectkbest', 'agglomeration', 'autoencoder'
MODEL_TYPE = 'stacked_ensemble'   # Options: 'elasticnet', 'xgboost', 'transformer', 'tabnet', 'gnn', 'ngboost', 'stacked_ensemble', etc.

# -- General Pipeline Params --
USE_HORVATH = True
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -- Feature Extractor Params --
PRE_FILTER_K = 20000              # Used to prevent OOM errors before Agglomeration/Autoencoder
FEATURE_K = 1000                  # Number of features to keep for SelectKBest
N_REGIONS = 500                   # Number of clusters for FeatureAgglomeration
ENCODING_DIM = 128                # Latent space dimension for Autoencoder

# -- Model Hyperparameters --
MODEL_PARAMS = {
    'elasticnet':   {'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0], 'cv': 5, 'max_iter': 10000},
    'randomforest': {'n_estimators': 100, 'max_depth': None},
    'xgboost':      {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 4},
    'catboost':     {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6},
    'ngboost':      {'n_estimators': 500, 'learning_rate': 0.01},
    'dnn':          {'hidden_layer_sizes': (512, 256, 128), 'alpha': 0.01, 'max_iter': 500},
    'gnn':          {'correlation_threshold': 0.6, 'epochs': 150, 'batch_size': 16, 'lr': 0.001},
    'transformer':  {'epochs': 50, 'batch_size': 16, 'lr': 0.001},
    'tabnet':       {'max_epochs': 50, 'patience': 10, 'batch_size': 16, 'virtual_batch_size': 8, 'lr': 1e-2}
}

# Define plotting prefix based on extractor
if FEATURE_EXTRACTOR == 'selectkbest': EXT_SUFFIX = f"k{FEATURE_K}"
elif FEATURE_EXTRACTOR == 'agglomeration': EXT_SUFFIX = f"n{N_REGIONS}"
else: EXT_SUFFIX = f"enc{ENCODING_DIM}"

PLOT_DIR = f"results/{DATASET}/{MODEL_TYPE}_{FEATURE_EXTRACTOR}"
os.makedirs(PLOT_DIR, exist_ok=True)


def horvath_transform(age, adult_age=20):
    if not USE_HORVATH: return age
    age = np.array(age)
    return np.where(age <= adult_age, np.log(age+1)-np.log(adult_age + 1), (age - adult_age)/(adult_age+1))

def horvath_inverse(transformed_age, adult_age=20):
    if not USE_HORVATH: return transformed_age
    transformed_age = np.array(transformed_age)
    return np.where(transformed_age < 0, np.exp(transformed_age + np.log(adult_age+1))-1, 
                    transformed_age * (adult_age+1) + adult_age)


print(f"--- Loading Dataset: {DATASET.upper()} ---")

if DATASET == 'bhak':
    metadata = pd.read_csv('meta_bhak.csv', sep=';', index_col=0)
    metadata.index = metadata.index.astype(str).str.strip()
    df = pd.read_csv('train_multi_omics_bhak.csv', sep=';', index_col=0)
    X_full = df.T
    target_col = 'Age (years)'

elif DATASET in ['temporal_pole', 'orb_cortex']:
    meta_file = f"meta_{DATASET}.csv"
    data_file = f"train_methylation_{DATASET}.csv"
    
    sep_meta = ',' if DATASET == 'temporal_pole' else ';'
    target_col = 'Age' if DATASET == 'temporal_pole' else 'age'

    metadata = pd.read_csv(meta_file, sep=sep_meta, index_col=0)
    if DATASET == 'temporal_pole':
        metadata.set_index('Sample', inplace=True)
    metadata.index = metadata.index.astype(str).str.replace('"', '').str.strip()

    print("Parsing massive CSV line-by-line to bypass memory limits...")
    with open(data_file, 'r') as f:
        header = f.readline().strip().replace('"', '').split(';')
        cpg_sites = header[1:]
        sample_ids, data_rows = [], []
        
        for i, line in enumerate(f):
            row_values = line.strip().replace('"', '').split(';')
            sample_ids.append(row_values[0].strip())
            data_rows.append(np.array(row_values[1:], dtype=np.float32))
            if (i + 1) % 10 == 0: print(f"Loaded patient {i+1} into RAM...")

    X_full = pd.DataFrame(np.vstack(data_rows), index=sample_ids, columns=cpg_sites)

X_full.index = X_full.index.astype(str).str.strip()
samples = X_full.index.intersection(metadata.index)
print(f"SUCCESS: Found {len(samples)} matching patients!")

X = X_full.loc[samples].astype('float32')
y = metadata.loc[samples, target_col]

y_transformed = horvath_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE)


print(f"--- Extracting Features via: {FEATURE_EXTRACTOR.upper()} ---")

feature_names = None

if FEATURE_EXTRACTOR == 'selectkbest':
    print(f"Selecting top {FEATURE_K} features...")
    selector = SelectKBest(score_func=f_regression, k=FEATURE_K)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    feature_names = X.columns[selector.get_support()]
    X_test_reduced = X_test[feature_names]

elif FEATURE_EXTRACTOR == 'agglomeration':
    print(f"Pre-filtering to {PRE_FILTER_K} features before clustering...")
    selector = SelectKBest(score_func=f_regression, k=PRE_FILTER_K)
    X_train_filtered = selector.fit_transform(X_train, y_train)
    X_test_filtered = X_test.iloc[:, selector.get_support()]

    print(f"Clustering into {N_REGIONS} robust regions...")
    agglo = FeatureAgglomeration(n_clusters=N_REGIONS, metric='euclidean', linkage='ward')
    X_train_reduced = agglo.fit_transform(X_train_filtered)
    X_test_reduced = agglo.transform(X_test_filtered)

elif FEATURE_EXTRACTOR == 'autoencoder':
    print(f"Pre-filtering to {PRE_FILTER_K} features before compression...")
    selector = SelectKBest(score_func=f_regression, k=PRE_FILTER_K)
    X_train_filtered = selector.fit_transform(X_train, y_train)
    X_test_filtered = X_test.iloc[:, selector.get_support()]

    print(f"Building Autoencoder ({ENCODING_DIM} dimensions)...")
    input_layer = Input(shape=(X_train_filtered.shape[1],))
    encoded = Dense(512, activation='relu')(input_layer)
    encoded = Dense(256, activation='relu')(encoded)
    bottleneck = Dense(ENCODING_DIM, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(bottleneck)
    decoded = Dense(512, activation='relu')(decoded)
    output_layer = Dense(X_train_filtered.shape[1], activation='linear')(decoded) 

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train_filtered, X_train_filtered, epochs=20, batch_size=32, verbose=1)

    encoder = Model(inputs=input_layer, outputs=bottleneck)
    X_train_reduced = encoder.predict(X_train_filtered)
    X_test_reduced = encoder.predict(X_test_filtered)

X_train_reduced = pd.DataFrame(X_train_reduced)
X_test_reduced = pd.DataFrame(X_test_reduced)


class EpigeneticGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super(EpigeneticGNN, self).__init__()
        torch.manual_seed(RANDOM_STATE)
        self.conv1 = GCNConv(1, hidden_channels) 
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  
        x = self.bn(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.lin(x)

class GNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, correlation_threshold=0.6, epochs=150, batch_size=16, lr=0.001):
        self.correlation_threshold = correlation_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EpigeneticGNN().to(self.device)
        self.edge_index = None
        self.scaler = StandardScaler()

    def _build_graph(self, X):
        corr_matrix = np.corrcoef(X.T)
        edges = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    edges.extend([[i, j], [j, i]])
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _prepare_data(self, X, y=None):
        data_list = []
        for i in range(X.shape[0]):
            x = torch.tensor(X[i]).unsqueeze(1).float()
            target = torch.tensor([y[i]]).float() if y is not None else None
            data_list.append(Data(x=x, edge_index=self.edge_index, y=target))
        return data_list

    def fit(self, X, y):
        if hasattr(X, 'to_numpy'): X = X.to_numpy()
        if hasattr(y, 'to_numpy'): y = y.to_numpy()
        X = self.scaler.fit_transform(X)
        self._build_graph(X)
        
        train_data = self._prepare_data(X, y)
        loader = GeoDataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for data in loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index, data.batch)
                loss = criterion(out.view(-1), data.y) 
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if hasattr(X, 'to_numpy'): X = X.to_numpy()
        X = self.scaler.transform(X)
        self.model.eval()
        loader = GeoDataLoader(self._prepare_data(X), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                preds.extend(self.model(data.x, data.edge_index, data.batch).view(-1).cpu().numpy())
        return np.array(preds)


# --- B. TABULAR TRANSFORMER ---
class TabularTransformer(nn.Module):
    def __init__(self, num_features, dim=64, depth=3, heads=4, dropout=0.1):
        super().__init__()
        self.feature_embeddings = nn.Parameter(torch.randn(1, num_features, dim))
        self.feature_biases = nn.Parameter(torch.randn(1, num_features, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1) * self.feature_embeddings + self.feature_biases
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0, :])

class TransformerRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, num_features, epochs=100, batch_size=32, lr=1e-3):
        self.num_features = num_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TabularTransformer(num_features=self.num_features).to(self.device)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        if hasattr(X, 'to_numpy'): X = X.to_numpy()
        if hasattr(y, 'to_numpy'): y = y.to_numpy()
        X = self.scaler.fit_transform(X)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
        loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if hasattr(X, 'to_numpy'): X = X.to_numpy()
        X = self.scaler.transform(X)
        self.model.eval()
        loader = TorchDataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch_X in loader:
                preds.extend(self.model(batch_X[0].to(self.device)).cpu().numpy().flatten())
        return np.array(preds)


# --- C. TABNET WRAPPER ---
class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, max_epochs=50, patience=10, batch_size=16, virtual_batch_size=8, lr=1e-2):
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=1, n_shared=2,
                                     optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=self.lr),
                                     mask_type='entmax', verbose=0, seed=RANDOM_STATE)

    def fit(self, X, y):
        if hasattr(X, 'to_numpy'): X = X.to_numpy()
        if hasattr(y, 'to_numpy'): y = y.to_numpy()
        X_np = self.scaler.fit_transform(X).astype(np.float32)
        y_np = y.reshape(-1, 1).astype(np.float32)
        
        self.model.fit(X_train=X_np, y_train=y_np, eval_set=[(X_np, y_np)], eval_name=['train'], eval_metric=['mae'],
                       max_epochs=self.max_epochs, patience=self.patience, batch_size=self.batch_size, 
                       virtual_batch_size=self.virtual_batch_size)
        return self

    def predict(self, X):
        if hasattr(X, 'to_numpy'): X = X.to_numpy()
        X_np = self.scaler.transform(X).astype(np.float32)
        return self.model.predict(X_np).flatten()

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


print(f"--- Building & Training Model: {MODEL_TYPE.upper()} ---")

# Stacked Regressor Definition
base_estimators = [
    ('elasticnet', ElasticNetCV(**MODEL_PARAMS['elasticnet'], n_jobs=-1, random_state=RANDOM_STATE)),
    ('histgradient', HistGradientBoostingRegressor(max_iter=500, random_state=RANDOM_STATE)),
    ('dnn', make_pipeline(
        StandardScaler(), 
        MLPRegressor(**MODEL_PARAMS['dnn'], random_state=RANDOM_STATE)
    ))
]

models = {
    'elasticnet': ElasticNetCV(**MODEL_PARAMS['elasticnet'], n_jobs=-1, random_state=RANDOM_STATE),
    'randomforest': RandomForestRegressor(**MODEL_PARAMS['randomforest'], n_jobs=-1, random_state=RANDOM_STATE),
    'xgboost': XGBRegressor(**MODEL_PARAMS['xgboost'], random_state=RANDOM_STATE, n_jobs=-1),
    'catboost': CatBoostRegressor(**MODEL_PARAMS['catboost'], random_seed=RANDOM_STATE, verbose=0),
    'ngboost': NGBRegressor(
        Base=DecisionTreeRegressor(criterion='friedman_mse', max_depth=3),
        Dist=Normal, **MODEL_PARAMS['ngboost'], random_state=RANDOM_STATE, verbose=False
    ),
    'dnn': make_pipeline(StandardScaler(), MLPRegressor(**MODEL_PARAMS['dnn'], random_state=RANDOM_STATE)),
    'transformer': TransformerRegressor(num_features=X_train_reduced.shape[1], **MODEL_PARAMS['transformer']),
    'tabnet': TabNetWrapper(**MODEL_PARAMS['tabnet']),
    'gnn': GNNRegressor(**MODEL_PARAMS['gnn']),
    'stacked_ensemble': StackingRegressor(
        estimators=base_estimators,
        final_estimator=RidgeCV(cv=5),
        n_jobs=-1,
        passthrough=False 
    )
}

model = models[MODEL_TYPE]
model.fit(X_train_reduced, y_train)


print("--- Evaluating Metrics ---")

if MODEL_TYPE == 'ngboost':
    y_dists = model.pred_dist(X_test_reduced)
    preds_transformed = y_dists.mean()
    uncertainty_transformed = y_dists.std()
    uncertainty_years = uncertainty_transformed * (20 + 1) if USE_HORVATH else uncertainty_transformed
else:
    preds_transformed = model.predict(X_test_reduced)

if MODEL_TYPE == 'pls': preds_transformed = preds_transformed.flatten()

preds_years = horvath_inverse(preds_transformed)
actual_years = horvath_inverse(y_test)

metrics = {
    "MAE": mean_absolute_error(actual_years, preds_years),
    "MedianAE": median_absolute_error(actual_years, preds_years),
    "R2": r2_score(actual_years, preds_years),
    "Pearson_R": pearsonr(actual_years, preds_years)[0]
}

print("\nFinal Model Metrics:")
for k, v in metrics.items(): print(f"  {k}: {v:.4f}")


print(f"--- Generating Plots in {PLOT_DIR} ---")

# 1. Main Scatter Plot
plt.figure(figsize=(8, 8))
if MODEL_TYPE == 'ngboost':
    sorted_indices = np.argsort(actual_years)
    plt.errorbar(actual_years[sorted_indices], preds_years[sorted_indices], 
                 yerr=uncertainty_years[sorted_indices] * 1.96, 
                 fmt='o', color='teal', ecolor='lightgray', elinewidth=2, capsize=3, alpha=0.8)
else:
    plt.scatter(actual_years, preds_years, alpha=0.6, color='teal', edgecolors='white')

lims = [0, 100]
plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
plt.xlabel("Actual Age (Years)")
plt.ylabel("Predicted Age (Years)")
plt.title(f"{DATASET.capitalize()} | {MODEL_TYPE} ({FEATURE_EXTRACTOR})\nMAE: {metrics['MAE']:.2f} | R2: {metrics['R2']:.2f}")
plt.savefig(os.path.join(PLOT_DIR, f"scatter_{EXT_SUFFIX}.png"))
plt.close()

# 2. Residual Plot
residuals = actual_years - preds_years
plt.figure(figsize=(8, 4))
plt.scatter(actual_years, residuals, alpha=0.5, color='coral')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Actual Age")
plt.ylabel("Error (Years)")
plt.savefig(os.path.join(PLOT_DIR, f"residuals_{EXT_SUFFIX}.png"))
plt.close()

# 3. EAA Distribution
plt.figure()
sns.histplot(preds_years - actual_years, kde=True, color='skyblue')
plt.axvline(x=0, color='red', linestyle='--')
plt.title("Epigenetic Age Acceleration")
plt.savefig(os.path.join(PLOT_DIR, f"eaa_{EXT_SUFFIX}.png"))
plt.close()

# 4. Feature Importances (ONLY if SelectKBest was used)
if FEATURE_EXTRACTOR == 'selectkbest' and (hasattr(model, 'coef_') or hasattr(model, 'feature_importances_')):
    importances = np.array(model.coef_ if hasattr(model, 'coef_') else model.feature_importances_).flatten()
    indices = np.argsort(np.abs(importances))[-20:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(20), importances[indices], color='plum')
    plt.yticks(range(20), [feature_names[i] for i in indices])
    plt.title(f"Top 20 Predictive CpG Sites ({MODEL_TYPE})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"importance_{EXT_SUFFIX}.png"))
    plt.close()


# SHAP requires 1-to-1 feature mapping, making it optimal for SelectKBest features.
if FEATURE_EXTRACTOR == 'selectkbest':
    print("--- Calculating SHAP Values ---")

    required_evals = (2 * X_test_reduced.shape[1]) + 1
    
    # Tree Models (Extremely Fast)
    if MODEL_TYPE in ['randomforest', 'extratrees', 'histgradient', 'catboost', 'xgboost']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_reduced.iloc[:20, :], max_evals=required_evals)

    # Linear Models (Fast, uses analytical solutions)
    elif MODEL_TYPE in ['elasticnet', 'ridge', 'lasso', 'bayesian_ridge', 'ard', 'linearsvr']:
        explainer = shap.LinearExplainer(model, X_train_reduced)
        shap_values = explainer(X_test_reduced.iloc[:20, :], max_evals=required_evals)

    # Pipelines & Complex Math (DNN, Ensembles, PyTorch, NGBoost)
    else:
        print("Using Permutation Explainer for pipelines and complex architectures. This might take a while...")
        background = shap.sample(X_train_reduced, 50)
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X_test_reduced.iloc[:20, :], max_evals=required_evals)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_reduced.iloc[:20, :], show=False)
    plt.title("SHAP Summary: Global Epigenetic Drivers of Aging")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"SHAP_summary_{EXT_SUFFIX}.png"), bbox_inches='tight')
    plt.close()

    # SHAP Waterfall Plot (for Patient 0)
    patient_index = 0 
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[patient_index], max_display=10, show=False)
    plt.title(f"Patient {patient_index} Prediction Breakdown (Actual Age: {actual_years[patient_index]:.1f})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"SHAP_waterfall_{EXT_SUFFIX}.png"), bbox_inches='tight')
    plt.close()

print("\nPipeline Complete! All assets saved.")
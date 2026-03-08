import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.spatial.distance import directed_hausdorff, cdist
from kan import KAN
import os
import shutil
import random

# ==========================================
# 1. Environment Setup and Directory Management
# ==========================================
def setup_env():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Ready: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CPU Only: No GPU detected")
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    base_dir = 'result_v6_data_only'
    
    sub_dirs = [
        'models',           # save trained model checkpoints
        'metrics',          # save evaluation metrics and training logs
        'cleaned_data',     # save cleaned datasets after geometric filtering
        'prediction_data'   # save predicted and true coordinate data for analysis
    ]
    
    if os.path.exists(base_dir):
        try:
            shutil.rmtree(base_dir)
        except Exception as e:
            print(f"Cannot delete old directory: {e}")
    
    for sub in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
        
    return device, base_dir

# ==========================================
# 2. Data Cleaning
# ==========================================
def clean_dataset_geometric(X, Y_contain, Y_diff):
    print("\n[1/7] Data Cleaning...")
    n_original = len(X)
    mid = 800
    
    lengths = []
    max_jumps = []
    
    for i in range(n_original):
        x = Y_contain[i, :mid]
        y = Y_contain[i, mid:]
        dx = np.diff(x)
        dy = np.diff(y)
        step_dists = np.sqrt(dx**2 + dy**2)
        lengths.append(np.sum(step_dists))
        max_jumps.append(np.max(step_dists) if len(step_dists) > 0 else 0)

    lengths = np.array(lengths)
    max_jumps = np.array(max_jumps)
    
    median_len = np.median(lengths)
    len_mask = (lengths > 0.6 * median_len) & (lengths < 1.4 * median_len)
    jump_mask = max_jumps < (0.05 * median_len)
    
    good_mask = len_mask & jump_mask
    valid_indices = np.where(good_mask)[0]
    
    print(f"   Original Data: {n_original}")
    print(f"   Remaining Valid: {len(valid_indices)}")
    
    if len(valid_indices) == 0:
        raise ValueError("[Error] All data has been filtered out!")
    return X[valid_indices], Y_contain[valid_indices], Y_diff[valid_indices], valid_indices

# ==========================================
# 3. Scientific Sampling using K-Means Clustering
# ==========================================
def select_scientific_samples(X_scaled, X_original, n_select, result_dir):
    print(f"   Executing sampling Confidence Data (K-Means, k={n_select})...")
    
    kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)
    train_indices = np.unique(closest_indices)
    
    if len(train_indices) < n_select:
        remaining = list(set(range(len(X_scaled))) - set(train_indices))
        needed = n_select - len(train_indices)
        fillers = np.random.choice(remaining, needed, replace=False)
        train_indices = np.concatenate([train_indices, fillers])
    
    all_indices = np.arange(len(X_scaled))
    test_indices = np.setdiff1d(all_indices, train_indices)
    
    print(f"   Selected {len(train_indices)} data for training")
    
    feature_names = ['AoA', 'TIME', 'v', 'h', 'mvc', 't0', 'LWC']
    train_phys_data = X_original[train_indices]
    stats_df = pd.DataFrame(train_phys_data, columns=feature_names)
    stats_summary = stats_df.describe().T[['min', 'max', 'mean', 'std']]
    
    save_path = os.path.join(result_dir, 'metrics', 'training_set_ranges.csv')
    stats_summary.to_csv(save_path)
    print(f"   Training set physical parameter ranges saved to: {save_path}")
    
    return train_indices, test_indices

# ==========================================
# 4. Calculation of Evaluation Metrics (MSE, MAE, R², IoU, Hausdorff, Chamfer)
# ==========================================
def calculate_chamfer_distance(t_x, t_y, p_x, p_y):
    true_pts = np.column_stack((t_x, t_y))
    pred_pts = np.column_stack((p_x, p_y))
    d = cdist(true_pts, pred_pts, 'euclidean')
    d1 = np.mean(np.min(d, axis=1))
    d2 = np.mean(np.min(d, axis=0))
    return (d1 + d2) / 2.0

def calculate_iou_raster(t_x, t_y, p_x, p_y, grid_size=60):
    from matplotlib.path import Path 
    x_min = min(np.min(t_x), np.min(p_x))
    x_max = max(np.max(t_x), np.max(p_x))
    y_min = min(np.min(t_y), np.min(p_y))
    y_max = max(np.max(t_y), np.max(p_y))
    
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    path_true = Path(np.column_stack((t_x, t_y)))
    path_pred = Path(np.column_stack((p_x, p_y)))
    mask_true = path_true.contains_points(points)
    mask_pred = path_pred.contains_points(points)
    intersection = np.sum(mask_true & mask_pred)
    union = np.sum(mask_true | mask_pred)
    return intersection / union if union != 0 else 0.0

def calculate_all_metrics(true_matrix, pred_matrix):
    n = true_matrix.shape[0]
    mid = 800
    metrics = {'mse': [], 'mae': [], 'r2': [], 'iou': [], 'hausdorff': [], 'chamfer': []}
    
    print("   Calculating all metrics...")
    for i in range(n):
        t_x, t_y = true_matrix[i, :mid], true_matrix[i, mid:]
        p_x, p_y = pred_matrix[i, :mid], pred_matrix[i, mid:]
        
        metrics['mse'].append(np.mean((true_matrix[i] - pred_matrix[i])**2))
        metrics['mae'].append(mean_absolute_error(true_matrix[i], pred_matrix[i]))
        metrics['r2'].append(r2_score(true_matrix[i], pred_matrix[i]))
        
        u = np.column_stack((t_x, t_y))
        v = np.column_stack((p_x, p_y))
        d1 = directed_hausdorff(u, v)[0]
        d2 = directed_hausdorff(v, u)[0]
        metrics['hausdorff'].append(max(d1, d2))
        metrics['chamfer'].append(calculate_chamfer_distance(t_x, t_y, p_x, p_y))
        metrics['iou'].append(calculate_iou_raster(t_x, t_y, p_x, p_y))
        
        if (i+1) % 1000 == 0: print(f"   Processed {i+1}/{n}")
        
    return pd.DataFrame(metrics)

# ==========================================
# 5. Data Loading, Cleaning, PCA, and Scientific Splitting
# ==========================================
def load_data(device, result_dir, data_dir='data'):
    print(f"\n[2/7] Data Loading and Scientific Splitting...")
    try:
        if not os.path.exists(data_dir): 
            if os.path.exists(os.path.join('kan_ice', 'data')):
                data_dir = os.path.join('kan_ice', 'data')
            else:
                data_dir = '.'
                
        df_input = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        df_contain = pd.read_csv(os.path.join(data_dir, 'containMatrix.csv'), header=None)
        df_diff = pd.read_csv(os.path.join(data_dir, 'diffMatrix.csv'), header=None)
    except Exception as e:
        print(f"Data Reading Error: {e}")
        exit()

    X_clean, Y_contain, Y_diff, valid_ids = clean_dataset_geometric(
        df_input.values, df_contain.values, df_diff.values
    )
    
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_clean)
    
    print("   Executing PCA...")
    pca = PCA(n_components=0.999)
    Y_pca = pca.fit_transform(Y_diff)
    n_out = Y_pca.shape[1]
    
    print(f"   Dimensions after PCA: {n_out}")
    
    n_total = len(X_clean)
    n_train_target = min(800, int(n_total * 0.9))
    
    idx_train, idx_test = select_scientific_samples(X_scaled, X_clean, n_train_target, result_dir)
    
    dataset = {
        'train_input': torch.from_numpy(X_scaled[idx_train]).float().to(device),
        'train_label': torch.from_numpy(Y_pca[idx_train]).float().to(device),
        'test_input': torch.from_numpy(X_scaled[idx_test]).float().to(device),
        'test_label': torch.from_numpy(Y_pca[idx_test]).float().to(device),
        'all_input': torch.from_numpy(X_scaled).float().to(device)
    }
    
    return dataset, pca, Y_contain, Y_diff, idx_train, idx_test, n_out, valid_ids

# ==========================================
# 6. Training the KAN Model
# ==========================================
def train_model(dataset, n_in, n_out, device, result_dir):
    print(f"\n[3/7] Model Training...")
    model = KAN(width=[n_in, 18, 36, n_out], grid=3, k=3, seed=42, device=device)
    
    results = model.fit(dataset, opt="LBFGS", steps=80, lamb=0.1, lamb_entropy=20.0)
    
    loss_df = pd.DataFrame({
        'train_loss': results['train_loss'],
        'test_loss': results['test_loss']
    })
    loss_df.to_csv(os.path.join(result_dir, 'metrics', 'loss_history.csv'), index=False)
    
    model_save_path = os.path.join(result_dir, 'models', 'kan_model_final.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"   Model saved to: {model_save_path}")
    
    return model

# ==========================================
# 7. Evaluation and Saving Results
# ==========================================
def evaluate_and_save_data(model, dataset, pca, Y_contain, Y_diff, result_dir, valid_ids, idx_train, idx_test):
    print(f"\n[4/7] Inferring...")
    model.eval()
    with torch.no_grad():
        pred_pca = model(dataset['all_input']).cpu().numpy()
        
    pred_diff = pca.inverse_transform(pred_pca)
    clean_wing = Y_contain - Y_diff
    pred_ice = clean_wing + pred_diff
    true_ice = Y_contain
    
    df_metrics = calculate_all_metrics(true_ice, pred_ice)
    df_metrics['original_id'] = valid_ids
    df_metrics['split'] = 'unknown'
    df_metrics.loc[idx_train, 'split'] = 'train'
    df_metrics.loc[idx_test, 'split'] = 'test'
    
    df_metrics.to_csv(os.path.join(result_dir, 'metrics', 'all_metrics.csv'), index=False)
    print(f"   Evaluation metrics saved")

    print(f"\n[5/7] Saving trajectory coordinate data to CSV...")
    save_path = os.path.join(result_dir, 'prediction_data')
    
    print("   Saving predicted shapes (pred_shapes.csv)...")
    df_pred = pd.DataFrame(pred_ice)
    df_pred.insert(0, 'original_id', valid_ids)
    df_pred.to_csv(os.path.join(save_path, 'pred_shapes.csv'), index=False)
    
    print("   Saving true shapes (true_shapes.csv)...")
    df_true = pd.DataFrame(true_ice)
    df_true.insert(0, 'original_id', valid_ids)
    df_true.to_csv(os.path.join(save_path, 'true_shapes.csv'), index=False)
    
    print("   Saving clean shapes (clean_shapes.csv)...")
    df_clean = pd.DataFrame(clean_wing)
    df_clean.insert(0, 'original_id', valid_ids)
    df_clean.to_csv(os.path.join(save_path, 'clean_shapes.csv'), index=False)
    
    print(f"   All trajectory data saved to: {save_path}")

if __name__ == "__main__":
    device, res_dir = setup_env()
    dataset, pca, Y_contain, Y_diff, idx_train, idx_test, n_out, valid_ids = load_data(device, res_dir)
    model = train_model(dataset, n_in=7, n_out=n_out, device=device, result_dir=res_dir)
    evaluate_and_save_data(model, dataset, pca, Y_contain, Y_diff, res_dir, valid_ids, idx_train, idx_test)
    
    print(f"\n[7/7] All done")
    print(f"      - Model: {res_dir}/models/kan_model_final.pth")
    print(f"      - Coordinates: {res_dir}/prediction_data/")
    print(f"      - Metrics: {res_dir}/metrics/")
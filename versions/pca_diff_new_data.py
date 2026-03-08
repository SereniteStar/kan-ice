import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from kan import KAN
import os
import shutil

# ==========================================
# 1. 配置与环境检查
# ==========================================
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 成功检测到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ 未检测到 GPU，将使用 CPU 运行。")
    return device

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. 数据加载与预处理
# ==========================================
def load_and_process_data(device, data_dir='new_data'):
    print(f"\n[1/7] 正在加载新数据 (路径: {data_dir})...")
    
    try:
        if not os.path.exists(data_dir):
            print(f"❌ 错误: 找不到文件夹 '{data_dir}'。")
            exit()
            
        # 1. 加载输入参数
        df_input = pd.read_csv(os.path.join(data_dir, 'data.csv'), encoding='gb18030').fillna(0)
        sample_ids = df_input.iloc[:, 0].values.astype(str) 
        X_raw = df_input.iloc[:, 1:].values.astype(float)
        
        # 剔除方差为0的列（常数特征）
        selector = np.var(X_raw, axis=0) > 1e-6
        if not np.all(selector):
            print(f"⚠️ 检测到 {np.sum(~selector)} 个常数特征列，已自动剔除。")
            X_raw = X_raw[:, selector]
        
        print(f"   有效输入特征维度: {X_raw.shape}")

        # 2. 加载几何数据
        df_clean = pd.read_csv(os.path.join(data_dir, 'cleanXY.csv'), header=None, encoding='gb18030').fillna(0)
        df_ice = pd.read_csv(os.path.join(data_dir, 'iceXY.csv'), header=None, encoding='gb18030').fillna(0)
        df_delta = pd.read_csv(os.path.join(data_dir, 'deltaXY.csv'), header=None, encoding='gb18030').fillna(0)
        
        Y_clean = df_clean.values
        Y_ice = df_ice.values
        Y_delta = df_delta.values 
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        exit()

    # 3. 输入归一化 (X -> [-1, 1])
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_x.fit_transform(X_raw)

    # 4. PCA 降维
    print("   正在执行 PCA 降维...")
    pca = PCA(n_components=0.995)
    Y_pca = pca.fit_transform(Y_delta)
    n_components = Y_pca.shape[1]
    print(f"   PCA 降维完成: {Y_delta.shape[1]} 维 -> {n_components} 维")

    # 5. 目标值归一化 (Y_pca -> [-1, 1])
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    Y_pca_scaled = scaler_y.fit_transform(Y_pca)

    # 6. 数据集划分
    all_indices = np.arange(len(X_raw))
    train_idx, test_idx = train_test_split(all_indices, test_size=50, random_state=42, shuffle=True)
    
    print(f"   数据集划分: 训练集 {len(train_idx)} 条, 测试集 {len(test_idx)} 条")

    X_train = X_scaled[train_idx]
    y_train = Y_pca_scaled[train_idx]
    X_test = X_scaled[test_idx]
    y_test = Y_pca_scaled[test_idx]

    dataset = {
        'train_input': torch.from_numpy(X_train).float().to(device),
        'train_label': torch.from_numpy(y_train).float().to(device),
        'test_input': torch.from_numpy(X_test).float().to(device),
        'test_label': torch.from_numpy(y_test).float().to(device)
    }

    aux_data = {
        'test_idx': test_idx,
        'sample_ids': sample_ids,
        'Y_clean': Y_clean,
        'Y_ice': Y_ice,
        'Y_delta': Y_delta
    }

    return dataset, pca, scaler_x, scaler_y, aux_data, n_components, X_raw.shape[1]

# ==========================================
# 3. KAN 模型训练
# ==========================================
def train_kan(dataset, n_input, n_output, device):
    print("\n[2/7] 初始化 KAN 模型...")
    
    model = KAN(width=[n_input, 10, n_output], grid=3, k=3, seed=42, device=device)

    print(f"   模型结构: [{n_input} -> 10 -> {n_output}]")
    print("   开始训练 (Adam, 无正则化)...")

    results = model.fit(dataset, opt="Adam", steps=200, lr=0.005, lamb=0.0, lamb_entropy=0.0)
    
    print("   训练完成。")
    return model, results

# ==========================================
# 4. 结果评估与重构
# ==========================================
def evaluate_and_reconstruct(model, dataset, pca, scaler_y, aux_data):
    print("\n[3/7] 正在评估与重构曲线...")
    
    model.eval()
    with torch.no_grad():
        pred_pca_scaled_tensor = model(dataset['test_input'])
    
    pred_pca_scaled = pred_pca_scaled_tensor.cpu().numpy()
    
    if np.isnan(pred_pca_scaled).any():
        print("❌ 警告: 预测结果包含 NaN，替换为 0。")
        pred_pca_scaled = np.nan_to_num(pred_pca_scaled)

    # 反归一化 + PCA 反变换
    pred_pca = scaler_y.inverse_transform(pred_pca_scaled)
    pred_delta = pca.inverse_transform(pred_pca)

    test_indices = aux_data['test_idx']
    true_clean = aux_data['Y_clean'][test_indices]
    true_ice = aux_data['Y_ice'][test_indices]
    test_ids = aux_data['sample_ids'][test_indices]

    pred_ice = true_clean + pred_delta

    mse_per_sample = np.mean((pred_ice - true_ice)**2, axis=1)
    
    best_idx_local = np.argmin(mse_per_sample)
    worst_idx_local = np.argmax(mse_per_sample)
    
    print(f"   最佳样本 ID: {test_ids[best_idx_local]} (MSE: {mse_per_sample[best_idx_local]:.6f})")
    print(f"   最差样本 ID: {test_ids[worst_idx_local]} (MSE: {mse_per_sample[worst_idx_local]:.6f})")

    return {
        'mse_list': mse_per_sample,
        'best_idx': best_idx_local,
        'worst_idx': worst_idx_local,
        'clean_wing': true_clean,
        'true_ice': true_ice,
        'pred_ice': pred_ice,
        'sample_ids': test_ids
    }

# ==========================================
# 5. 绘图可视化 (统计图)
# ==========================================
def plot_summary_results(results_train, eval_data):
    print("\n[4/7] 正在绘制统计图像...")
    plt.rcdefaults()
    
    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(results_train['train_loss'], label='Train Loss')
    plt.plot(results_train['test_loss'], label='Test Loss')
    plt.title('Training Loss Curve')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('summary_loss.png', dpi=150)
    plt.close()

    # MSE
    plt.figure(figsize=(10, 6))
    plt.plot(eval_data['mse_list'], label='MSE per Sample')
    plt.title('MSE Curve on Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('summary_mse.png', dpi=150)
    plt.close()

# ==========================================
# 6. 保存全景结果 (result 文件夹)
# ==========================================
def save_full_results(eval_data, save_dir='result'):
    print(f"\n[5/7] 正在保存全景结果到 '{save_dir}'...")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    num_samples = len(eval_data['sample_ids'])
    mid = 280 
    plt.rcdefaults()

    for i in range(num_samples):
        sample_id = eval_data['sample_ids'][i]
        mse = eval_data['mse_list'][i]
        clean = eval_data['clean_wing'][i]
        true = eval_data['true_ice'][i]
        pred = eval_data['pred_ice'][i]
        
        plt.figure(figsize=(10, 8))
        plt.plot(clean[:mid], clean[mid:], 'r-', linewidth=2, label='Clean Airfoil')
        plt.plot(true[:mid], true[mid:], 'b.', markersize=3, alpha=0.5, label='True Ice')
        plt.plot(pred[:mid], pred[mid:], color='orange', marker='.', linestyle='None', markersize=4, alpha=0.8, label='Pred Ice')
        
        for k in range(0, mid, 10):
            plt.plot([true[k], pred[k]], [true[mid+k], pred[mid+k]], color='gray', alpha=0.3, linewidth=0.5)
            
        plt.title(f"Model: {sample_id} | MSE: {mse:.6f}")
        plt.legend(loc='upper left')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(os.path.join(save_dir, f"{sample_id}_full.png"), dpi=100)
        plt.close()

# ==========================================
# 7. 保存局部放大结果 (result_top 文件夹)
# ==========================================
def save_zoomed_results(eval_data, save_dir='result_top'):
    print(f"\n[6/7] 正在保存局部放大结果到 '{save_dir}'...")
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    num_samples = len(eval_data['sample_ids'])
    mid = 280 
    plt.rcdefaults()

    for i in range(num_samples):
        sample_id = eval_data['sample_ids'][i]
        mse = eval_data['mse_list'][i]
        clean = eval_data['clean_wing'][i]
        true = eval_data['true_ice'][i]
        pred = eval_data['pred_ice'][i]
        
        plt.figure(figsize=(10, 8))
        
        # 绘图逻辑相同
        plt.plot(clean[:mid], clean[mid:], 'r-', linewidth=2, label='Clean Airfoil')
        plt.plot(true[:mid], true[mid:], 'b.', markersize=4, alpha=0.6, label='True Ice') # 点稍微大一点
        plt.plot(pred[:mid], pred[mid:], color='orange', marker='.', linestyle='None', markersize=5, alpha=0.8, label='Pred Ice')
        
        # 误差线
        for k in range(0, mid, 5): # 局部图可以画密一点
            plt.plot([true[k], pred[k]], [true[mid+k], pred[mid+k]], color='gray', alpha=0.3, linewidth=0.5)
            
        plt.title(f"Model: {sample_id} (Zoomed) | MSE: {mse:.6f}")
        plt.legend(loc='upper left')
        
        # 【关键修改】设置坐标轴范围，聚焦机翼头部
        # X轴：从 -0.1 (冰角) 到 0.3 (机翼前段)
        # Y轴：从 -0.15 到 0.15 (覆盖大部分结冰厚度)
        plt.xlim([-0.1, 0.3])
        plt.ylim([-0.15, 0.15])
        
        # 保持比例一致，防止变形
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(os.path.join(save_dir, f"{sample_id}_top.png"), dpi=100)
        plt.close()
        
        if (i + 1) % 10 == 0:
            print(f"   已保存 {i + 1}/{num_samples} 张...")

    print(f"   ✅ 全部 {num_samples} 张局部放大图已保存至 {save_dir}/")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    device = setup_device()
    dataset, pca, scaler_x, scaler_y, aux_data, n_out, n_in_real = load_and_process_data(device, data_dir='new_data')
    
    model, results = train_kan(dataset, n_input=n_in_real, n_output=n_out, device=device)
    
    eval_data = evaluate_and_reconstruct(model, dataset, pca, scaler_y, aux_data)
    
    plot_summary_results(results, eval_data)
    save_full_results(eval_data, save_dir='result')
    save_zoomed_results(eval_data, save_dir='result_top') # 新增调用
    
    print("\n[7/7] 🎉 程序执行完毕。")
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. 数据加载与预处理 (前800训练，后200测试)
# ==========================================
def load_and_process_data(device, data_dir='data'):
    print(f"\n[1/6] 正在加载数据 (路径: {data_dir})...")
    
    try:
        if not os.path.exists(data_dir):
            data_dir = os.path.join('kan_ice', 'data')
            
        df_input = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        df_contain = pd.read_csv(os.path.join(data_dir, 'containMatrix.csv'), header=None)
        df_diff = pd.read_csv(os.path.join(data_dir, 'diffMatrix.csv'), header=None)
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到数据文件。\n详细信息: {e}")
        exit()

    # 1. 输入特征 (X)
    X_raw = df_input.values
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)

    # 2. 目标特征 (Y - Diff)
    Y_diff = df_diff.values

    # 3. PCA 降维
    print("   正在执行 PCA 降维...")
    pca = PCA(n_components=0.995)
    Y_pca = pca.fit_transform(Y_diff)
    n_components = Y_pca.shape[1]
    print(f"   PCA 降维完成: 1600 维 -> {n_components} 维")

    # 4. 数据集划分 (前800训练，后200测试)
    train_size = 800
    test_size = 200
    
    total_samples = len(X_raw)
    if total_samples < (train_size + test_size):
        print(f"⚠️ 警告: 数据总量 ({total_samples}) 不足，将使用剩余数据作为测试集。")
        test_size = total_samples - train_size

    print(f"   数据集划分: 训练集 {train_size} 条, 测试集 {test_size} 条")

    X_train = X_scaled[:train_size]
    y_train = Y_pca[:train_size]
    
    X_test = X_scaled[train_size : train_size + test_size]
    y_test = Y_pca[train_size : train_size + test_size]
    
    idx_test = np.arange(train_size, train_size + test_size)

    # 5. 转换为 Tensor 并移动到 GPU
    print(f"   正在将数据移动到设备: {device}...")
    dataset = {
        'train_input': torch.from_numpy(X_train).float().to(device),
        'train_label': torch.from_numpy(y_train).float().to(device),
        'test_input': torch.from_numpy(X_test).float().to(device),
        'test_label': torch.from_numpy(y_test).float().to(device)
    }

    return dataset, pca, scaler_x, df_contain.values, Y_diff, idx_test, n_components

# ==========================================
# 3. KAN 模型训练
# ==========================================
def train_kan(dataset, n_input, n_output, device):
    print("\n[2/6] 初始化 KAN 模型...")
    
    model = KAN(width=[n_input, 10, n_output], grid=5, k=3, seed=42, device=device)

    print(f"   模型结构: [7 -> 10 -> {n_output}]")
    print("   开始训练 (LBFGS)...")

    results = model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001, lamb_entropy=2.0)
    
    print("   训练完成。")
    return model, results

# ==========================================
# 4. 结果评估与重构
# ==========================================
def evaluate_and_reconstruct(model, dataset, pca, Y_contain_all, Y_diff_all, idx_test):
    print("\n[3/6] 正在评估与重构曲线...")
    
    model.eval()
    with torch.no_grad():
        pred_pca_tensor = model(dataset['test_input'])
    
    pred_pca = pred_pca_tensor.cpu().numpy()
    pred_diff = pca.inverse_transform(pred_pca)

    true_ice = Y_contain_all[idx_test]
    true_diff = Y_diff_all[idx_test]
    clean_wing = true_ice - true_diff

    pred_ice = clean_wing + pred_diff

    mse_per_sample = np.mean((pred_ice - true_ice)**2, axis=1)
    
    best_idx = np.argmin(mse_per_sample)
    worst_idx = np.argmax(mse_per_sample)
    
    print(f"   最佳样本索引: {idx_test[best_idx]} (MSE: {mse_per_sample[best_idx]:.6f})")
    print(f"   最差样本索引: {idx_test[worst_idx]} (MSE: {mse_per_sample[worst_idx]:.6f})")

    return {
        'mse_list': mse_per_sample,
        'best_idx': best_idx,
        'worst_idx': worst_idx,
        'clean_wing': clean_wing,
        'true_ice': true_ice,
        'pred_ice': pred_ice,
        'sample_ids': idx_test
    }

# ==========================================
# 5. 绘图可视化 (统计图)
# ==========================================
def plot_summary_results(results_train, eval_data):
    print("\n[4/6] 正在绘制统计图像...")
    
    plt.rcdefaults() # 使用默认字体
    
    # Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(results_train['train_loss'], label='Train Loss')
    plt.plot(results_train['test_loss'], label='Test Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('summary_loss.png', dpi=150)
    plt.close()

    # MSE 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(eval_data['mse_list'], label='MSE per Sample')
    plt.title('MSE Curve on Test Set')
    plt.xlabel('Sample Index (0-199)')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('summary_mse.png', dpi=150)
    plt.close()

# ==========================================
# 6. 批量保存所有测试集结果 (新增功能)
# ==========================================
def save_all_test_results(eval_data, save_dir='result'):
    print(f"\n[5/6] 正在保存所有测试集结果到 '{save_dir}' 文件夹...")
    
    # 如果文件夹存在，先清空；不存在则创建
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    num_samples = len(eval_data['sample_ids'])
    mid = 800 # 数据分界点
    
    # 使用默认字体
    plt.rcdefaults()

    for i in range(num_samples):
        sample_id = eval_data['sample_ids'][i]
        mse = eval_data['mse_list'][i]
        
        clean = eval_data['clean_wing'][i]
        true = eval_data['true_ice'][i]
        pred = eval_data['pred_ice'][i]
        
        plt.figure(figsize=(10, 8))
        
        # 绘制
        plt.plot(clean[:mid], clean[mid:], 'r-', linewidth=2, label='Clean Airfoil')
        plt.plot(true[:mid], true[mid:], 'b.', markersize=2, alpha=0.5, label='True Ice')
        plt.plot(pred[:mid], pred[mid:], color='orange', marker='.', linestyle='None', markersize=3, alpha=0.8, label='Pred Ice')
        
        # 误差线 (每隔 20 点)
        for k in range(0, mid, 20):
            plt.plot([true[k], pred[k]], [true[mid+k], pred[mid+k]], color='gray', alpha=0.3, linewidth=0.5)
            
        plt.title(f"Sample {sample_id} | MSE: {mse:.6f}")
        plt.legend(loc='upper left')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 保存文件名格式: sample_0800.png
        filename = os.path.join(save_dir, f"sample_{sample_id:04d}.png")
        plt.savefig(filename, dpi=100) # dpi=100 保证速度，需要高清可调高
        plt.close()
        
        if (i + 1) % 20 == 0:
            print(f"   已保存 {i + 1}/{num_samples} 张...")

    print(f"   ✅ 全部 {num_samples} 张图片已保存至 {save_dir}/")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 设置设备
    device = setup_device()
    
    # 2. 加载数据
    dataset, pca, scaler, Y_contain_all, Y_diff_all, idx_test, n_out = load_and_process_data(device, data_dir='data')
    
    # 3. 训练模型
    model, results = train_kan(dataset, n_input=7, n_output=n_out, device=device)
    
    # 4. 评估
    eval_data = evaluate_and_reconstruct(model, dataset, pca, Y_contain_all, Y_diff_all, idx_test)
    
    # 5. 绘制统计图
    plot_summary_results(results, eval_data)
    
    # 6. 批量保存所有结果
    save_all_test_results(eval_data, save_dir='result')
    
    print("\n[6/6] 🎉 程序执行完毕。")
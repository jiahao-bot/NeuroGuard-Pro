import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import seaborn as sns
import os
import pandas as pd
import torch.nn.functional as F
import math

# ==============================================================================
# 🎨绘图风格配置
# ==============================================================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
GROUP_PALETTE = sns.color_palette(['#4c72b0', '#dd8452'])

EEG_POSITIONS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.3, 0.6), 'Fz': (0.0, 0.6), 'F4': (0.3, 0.6), 'F8': (0.7, 0.6),
    'T3': (-0.8, 0.0), 'C3': (-0.4, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.4, 0.0), 'T4': (0.8, 0.0),
    'T5': (-0.7, -0.6), 'P3': (-0.3, -0.6), 'Pz': (0.0, -0.6), 'P4': (0.3, -0.6), 'T6': (0.7, -0.6),
    'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
}
CHANNEL_NAMES = list(EEG_POSITIONS.keys())

# ==============================================================================
# 🎮 配置区域
# ==============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = '../processed_data/data_19ch.npz'
SAVE_DIR = '../results/Final_Paper_Visuals_V3'  # 新目录 V3

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

experiments = {
    # --- Random Group ---
    "Exp1_CNN_Random": ("CNN", "Random"),
    "Exp3_LSTM_random": ("LSTM", "Random"),
    "Exp5_Transformer_Random": ("Transformer", "Random"),
    "Exp6_GCN_Random": ("GCN", "Random"),
    "Exp7_DSS_GCN_Random": ("DSS-GCN", "Random"),
    "Exp8_Contrastive_Random": ("CV-GCN", "Random"),
    # --- Strict Group ---
    "Exp2_CNN_Strict": ("CNN", "Strict"),
    "Exp4_LSTM_Strict": ("LSTM", "Strict"),
    "Exp5_Transformer_Strict": ("Transformer", "Strict"),
    "Exp6_GCN_Strict": ("GCN", "Strict"),
    "Exp7_DualStream_Spectral_GCN_Strict": ("DSS-GCN", "Strict"),
    "Exp8_Contrastive_Consistency_SOTA_Strict": ("CV-GCN", "Strict")
}


# ==============================================================================
# 1. 模型定义 (保持不变)
# ==============================================================================
# --- LSTM ---
class BaselineLSTM(nn.Module):
    def __init__(self):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=19, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# --- CNN ---
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(19, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(128 * 64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2)
        )

    def forward(self, x): return self.net(x)


# --- Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x): return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv1d(19, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.pos_encoder = PositionalEncoding(d_model=64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.3,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 128, 128), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(128, 2))

    def forward(self, x):
        x = self.feature_extract(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x)


# --- GCN ---
class StandardGCN(nn.Module):
    def __init__(self):
        super(StandardGCN, self).__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(19, 19 * 8, 5, padding=2, groups=19), nn.BatchNorm1d(19 * 8), nn.ReLU(), nn.MaxPool1d(4)
        )
        self.adj = nn.Parameter(torch.rand(19, 19))
        self.gcn_weight = nn.Linear(8, 16)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(19 * 16 * 128, 128), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(128, 2))

    def forward(self, x):
        B = x.size(0)
        x = self.temporal_conv(x).view(B, 19, 8, -1)
        A = torch.softmax(self.adj, dim=1)
        support = torch.einsum('ij,bjft->bift', A, x)
        support = support.permute(0, 1, 3, 2)
        out = torch.relu(self.gcn_weight(support))
        return self.classifier(out)


# --- DSS-GCN ---
class DSS_SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(DSS_SpectralConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.MaxPool1d(2)
        )

    def forward(self, x): return self.conv(x)


class DSS_GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DSS_GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        support = torch.einsum('ij,bjf->bif', adj, x)
        out = self.fc(support)
        return self.act(out)


class DSS_GCN(nn.Module):
    def __init__(self):
        super(DSS_GCN, self).__init__()
        self.low_conv = DSS_SpectralConv(19, 19 * 16, 15, 19)
        self.adj_low = nn.Parameter(torch.rand(19, 19))
        self.gcn_low = DSS_GraphConvLayer(16, 32)
        self.high_conv = DSS_SpectralConv(19, 19 * 16, kernel_size=3, groups=19)
        self.adj_high = nn.Parameter(torch.rand(19, 19))
        self.gcn_high = DSS_GraphConvLayer(16, 32)
        self.fusion_fc = nn.Linear(32 * 2, 64)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(19 * 64, 64), nn.ReLU(),
                                        nn.Dropout(0.5), nn.Linear(64, 2))

    def forward(self, x):
        B = x.size(0)
        x_low = self.low_conv(x).view(B, 19, 16, -1).mean(dim=3)
        A_low = torch.softmax(self.adj_low + torch.eye(19).to(x.device), dim=1)
        out_low = self.gcn_low(x_low, A_low)
        x_high = self.high_conv(x).view(B, 19, 16, -1).mean(dim=3)
        A_high = torch.softmax(self.adj_high + torch.eye(19).to(x.device), dim=1)
        out_high = self.gcn_high(x_high, A_high)
        combined = torch.cat([out_low, out_high], dim=2)
        combined = torch.relu(self.fusion_fc(combined))
        return self.classifier(combined)


# --- CV-GCN ---
class CV_SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(CV_SpectralConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(0.3), nn.MaxPool1d(2)
        )

    def forward(self, x): return self.conv(x)


class CV_GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(CV_GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat)
        self.bn = nn.BatchNorm1d(19)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        support = torch.einsum('ij,bjf->bif', adj, x)
        out = self.fc(support)
        out = self.bn(out)
        return self.act(out)


class CV_GCN(nn.Module):
    def __init__(self):
        super(CV_GCN, self).__init__()
        self.low_conv = CV_SpectralConv(19, 19 * 16, 15, 19)
        self.adj_low = nn.Parameter(torch.randn(19, 19) * 0.01)
        self.gcn_low = CV_GraphConvLayer(16, 32)
        self.high_conv = CV_SpectralConv(19, 19 * 16, 3, 19)
        self.adj_high = nn.Parameter(torch.randn(19, 19) * 0.01)
        self.gcn_high = CV_GraphConvLayer(16, 32)
        self.proj_low = nn.Linear(19 * 32, 64)
        self.proj_high = nn.Linear(19 * 32, 64)
        self.fusion_fc = nn.Linear(32 * 2, 64)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.6), nn.Linear(19 * 64, 64), nn.ReLU(),
                                        nn.Dropout(0.6), nn.Linear(64, 2))

    def forward(self, x):
        B = x.size(0)
        x_low = self.low_conv(x).view(B, 19, 16, -1).mean(dim=3)
        A_low = torch.softmax(self.adj_low + torch.eye(19).to(x.device), dim=1)
        feat_low = self.gcn_low(x_low, A_low)
        x_high = self.high_conv(x).view(B, 19, 16, -1).mean(dim=3)
        A_high = torch.softmax(self.adj_high + torch.eye(19).to(x.device), dim=1)
        feat_high = self.gcn_high(x_high, A_high)
        combined = torch.cat([feat_low, feat_high], dim=2)
        combined = torch.relu(self.fusion_fc(combined))
        logits = self.classifier(combined)
        return logits, None, None, None, None


# ==============================================================================
# 🧠 核心绘图函数
# ==============================================================================
def plot_brain_network_2d(adj_matrix, importance_scores, model_name, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    head_circle = mpatches.Circle((0, 0), 1.0, color='lightgrey', fill=False, linewidth=2)
    ax.add_patch(head_circle)
    ax.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], color='lightgrey', linewidth=2)

    x_coords = [EEG_POSITIONS[ch][0] for ch in CHANNEL_NAMES]
    y_coords = [EEG_POSITIONS[ch][1] for ch in CHANNEL_NAMES]
    norm_importance = (importance_scores - importance_scores.min()) / (
                importance_scores.max() - importance_scores.min() + 1e-8)
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(norm_importance)
    sizes = 600 + 1200 * norm_importance

    threshold = np.percentile(adj_matrix, 80)
    for i in range(len(CHANNEL_NAMES)):
        for j in range(i + 1, len(CHANNEL_NAMES)):
            weight = adj_matrix[i, j]
            if weight > threshold:
                alpha = (weight - threshold) / (adj_matrix.max() - threshold + 1e-8)
                linewidth = 1.5 + 3.5 * alpha
                ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]],
                        color='darkred', alpha=alpha * 0.7, linewidth=linewidth, zorder=1)

    sc = ax.scatter(x_coords, y_coords, s=sizes, c=colors, edgecolors='black', linewidth=1.5, zorder=2)
    for i, txt in enumerate(CHANNEL_NAMES):
        ax.annotate(txt, (x_coords[i], y_coords[i]), xytext=(0, 0),
                    textcoords='offset points', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white' if norm_importance[i] > 0.6 else 'black')

    ax.set_xlim(-1.2, 1.2);
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal');
    ax.axis('off')
    plt.title(f"Learned Brain Network Topology ({model_name})", fontsize=18, pad=20)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label('Region Importance (Normalized)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 🚀 主执行逻辑
# ==============================================================================
def run():
    print("=" * 60)
    print("🚀 论文最终全能评估 (独立表格版)")
    print("=" * 60)

    data = np.load(DATA_PATH)
    X_test_raw, y_test = data['X_test'], data['y_test']
    X_train_raw = data['X_train']
    mean, std = np.mean(X_train_raw), np.std(X_train_raw)
    X_test = (X_test_raw - mean) / (std + 1e-8)

    test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE)),
                         batch_size=64, shuffle=False)

    results = []
    best_model_obj = None
    best_model_name = ""
    best_model_acc = 0.0

    for folder_name, (model_name, split_type) in experiments.items():
        print(f"\n🔍 评估: {model_name} [{split_type}]")
        path = os.path.join('../results/', folder_name, 'best_model.pth')

        if not os.path.exists(path): continue

        if model_name == "LSTM":
            model = BaselineLSTM()
        elif model_name == "CNN":
            model = BaselineCNN()
        elif model_name == "Transformer":
            model = TransformerModel()
        elif model_name == "GCN":
            model = StandardGCN()
        elif model_name == "DSS-GCN":
            model = DSS_GCN()
        elif model_name == "CV-GCN":
            model = CV_GCN()
        else:
            continue

        model = model.to(DEVICE)
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE), strict=True)
            model.eval()
        except:
            continue

        preds, labels, probs = [], [], []
        with torch.no_grad():
            for x, y in test_dl:
                out = model(x)
                if isinstance(out, tuple): out = out[0]
                prob = F.softmax(out, dim=1)[:, 1]
                _, p = torch.max(out, 1)
                preds.extend(p.cpu().numpy())
                labels.extend(y.cpu().numpy())
                probs.extend(prob.cpu().numpy())

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0

        print(f"   ✅ Acc: {acc * 100:.2f}%")
        results.append({'Model': model_name, 'Split': split_type,
                        'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC': auc})

        if split_type == "Strict" and model_name in ["DSS-GCN", "CV-GCN"]:
            if acc > best_model_acc:
                best_model_acc = acc
                best_model_obj = model
                best_model_name = model_name

    # --- 2. 生成分开的 CSV 表格 (Random / Strict) ---
    if len(results) > 0:
        df = pd.DataFrame(results)

        # 1. Random 表格
        df_random = df[df['Split'] == 'Random'].sort_values(by='Accuracy', ascending=False)
        # 重新排序列顺序
        df_random = df_random[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
        df_random.to_csv(os.path.join(SAVE_DIR, 'Ranking_Random_Split.csv'), index=False)

        # 2. Strict 表格
        df_strict = df[df['Split'] == 'Strict'].sort_values(by='Accuracy', ascending=False)
        df_strict = df_strict[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
        df_strict.to_csv(os.path.join(SAVE_DIR, 'Ranking_Strict_Split.csv'), index=False)

        print(f"\n📄 表格已生成:\n - {SAVE_DIR}/Ranking_Random_Split.csv\n - {SAVE_DIR}/Ranking_Strict_Split.csv")

        # --- 3. 绘制对比图 (纵向) ---
        print("\n📊 正在绘制对比图...")
        plot_df = df.melt(id_vars=['Model', 'Split'], value_vars=['Accuracy'], var_name='Metric', value_name='Score')
        plot_df['Score'] = plot_df['Score'] * 100

        # 按照 Strict Accuracy 排序模型
        model_order = df_strict['Model'].tolist()

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=plot_df, x='Model', y='Score', hue='Split',
                         order=model_order, palette=GROUP_PALETTE, edgecolor=".2")

        plt.title('Model Accuracy Comparison: Random vs. Subject-Independent Split', fontsize=16, pad=20,
                  fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Model Architecture', fontsize=14, fontweight='bold')
        plt.ylim(60, 105)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Data Split', title_fontsize='12', fontsize='11', loc='upper right')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3, fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'Final_Accuracy_Comparison_Plot_V3.png'), dpi=300)
        plt.close()
        print(f"✅ 对比图已生成: {SAVE_DIR}/Final_Accuracy_Comparison_Plot_V3.png")

    # --- 4. 脑图 (不变) ---
    if best_model_obj is not None:
        print(f"\n🧠 正在绘制最佳模型 ({best_model_name}) 的脑网络图...")
        if hasattr(best_model_obj, 'adj_high'):
            adj = best_model_obj.adj_high.detach().cpu().numpy()
        elif hasattr(best_model_obj, 'adj_low'):
            adj = best_model_obj.adj_low.detach().cpu().numpy()
        else:
            return

        adj_norm = (adj - adj.min()) / (adj.max() - adj.min())
        importance = np.sum(adj_norm, axis=1)

        df_imp = pd.DataFrame({'Region': CHANNEL_NAMES, 'Importance': importance}).sort_values(by='Importance',
                                                                                               ascending=False)
        txt_path = os.path.join(SAVE_DIR, 'Brain_Region_Ranking.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Brain Region Importance Ranking ({best_model_name})\n{'=' * 50}\nRank\tRegion\tScore\n")
            for i, row in df_imp.reset_index(drop=True).iterrows(): f.write(
                f"{i + 1}\t{row['Region']}\t{row['Importance']:.4f}\n")

        plot_brain_network_2d(adj_norm, importance, best_model_name,
                              os.path.join(SAVE_DIR, 'Final_Brain_Network_Topology.png'))
        print(f"✅ 脑图及排名已生成: {SAVE_DIR}")

    print(f"\n🎉 全部完成！请查看文件夹: {SAVE_DIR}")


if __name__ == '__main__':
    run()
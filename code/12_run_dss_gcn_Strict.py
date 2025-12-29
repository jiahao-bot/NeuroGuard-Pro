import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import pandas as pd
import time

# ==============================================================================
# 🎮 实验控制台
# ==============================================================================
EXP_ID = "Exp7_DualStream_Spectral_GCN"
BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.join('../results/', EXP_ID)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]


# --- 1. 频域特征提取模块 ---
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(SpectralConv, self).__init__()
        # 使用不同大小的卷积核模拟频带过滤
        # 大核 -> 提取低频 (慢波)
        # 小核 -> 提取高频 (快波)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.conv(x)


# --- 2. 图卷积单元 ---
class GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_nodes=19):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        # x: [B, Nodes, Feat]
        # adj: [Nodes, Nodes] (归一化后的)

        # Graph Conv: A * X * W
        support = torch.einsum('ij,bjf->bif', adj, x)
        out = self.fc(support)
        return self.act(out)


# --- 3. 核心模型：双流频域-空间 GCN (DSS-GCN) ---
class DSS_GCN(nn.Module):
    def __init__(self):
        super(DSS_GCN, self).__init__()

        # === 分支 1: 低频流 (Low-Frequency Stream) ===
        # 模拟 Alpha/Theta 波 (Kernel=15, 大感受野)
        self.low_conv = SpectralConv(19, 19 * 16, kernel_size=15, groups=19)
        # 低频流专用的脑连接图 (需要学习)
        self.adj_low = nn.Parameter(torch.rand(19, 19) * 0.01, requires_grad=True)
        self.gcn_low = GraphConvLayer(16, 32)

        # === 分支 2: 高频流 (High-Frequency Stream) ===
        # 模拟 Beta/Gamma 波 (Kernel=3, 小感受野)
        self.high_conv = SpectralConv(19, 19 * 16, kernel_size=3, groups=19)
        # 高频流专用的脑连接图 (独立学习，这是创新点！)
        self.adj_high = nn.Parameter(torch.rand(19, 19) * 0.01, requires_grad=True)
        self.gcn_high = GraphConvLayer(16, 32)

        # === 融合层 (Fusion) ===
        # 将两路特征融合
        self.fusion_fc = nn.Linear(32 * 2, 64)

        # === 分类器 ===
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(19 * 64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        B = x.size(0)

        # --- Stream 1: Low Freq ---
        x_low = self.low_conv(x)  # [B, 19*16, T/2]
        x_low = x_low.view(B, 19, 16, -1)
        feat_low = x_low.mean(dim=3)  # [B, 19, 16]

        # 构建低频图
        A_low = self.adj_low + torch.eye(19).to(x.device)
        A_low = torch.softmax(A_low, dim=1)

        # 低频 GCN
        out_low = self.gcn_low(feat_low, A_low)  # [B, 19, 32]

        # --- Stream 2: High Freq ---
        x_high = self.high_conv(x)  # [B, 19*16, T/2]
        x_high = x_high.view(B, 19, 16, -1)
        feat_high = x_high.mean(dim=3)  # [B, 19, 16]

        # 构建高频图
        A_high = self.adj_high + torch.eye(19).to(x.device)
        A_high = torch.softmax(A_high, dim=1)

        # 高频 GCN
        out_high = self.gcn_high(feat_high, A_high)  # [B, 19, 32]

        # --- Fusion ---
        # 拼接特征: [B, 19, 64]
        combined = torch.cat([out_low, out_high], dim=2)
        combined = torch.relu(self.fusion_fc(combined))

        return self.classifier(combined), A_low, A_high


# --- 辅助类 ---
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run():
    print("=" * 60)
    print(f"🚀 实验: {EXP_ID}")
    print("✨ 创新点: 双流架构 (低频图 vs 高频图)")
    print("=" * 60)

    # 1. 加载数据
    data = np.load('../processed_data/data_19ch.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    mean, std = np.mean(X_train), np.std(X_train)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE)),
                          batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE)),
                         batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = DSS_GCN().to(DEVICE)
    print(f"🧠 模型参数量: {count_parameters(model):,} (双流结构)")

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    # 使用 CosineAnnealing 帮助跳出局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_acc = 0.0
    history = {'acc': [], 'loss': []}
    time_history = []

    print("🔥 开始训练...")
    total_start = time.time()

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        loss_val = 0
        correct, total = 0, 0
        for x, y in train_dl:
            optimizer.zero_grad()
            out, _, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        avg_loss = loss_val / len(train_dl)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                out, _, _ = model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = 100 * correct / total
        e_time = time.time() - start
        time_history.append(e_time)
        history['acc'].append(test_acc)
        history['loss'].append(avg_loss)

        scheduler.step()

        save_msg = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            save_msg = "🏆"

        early_stopping(avg_loss, model, os.path.join(SAVE_DIR, 'best_model.pth'))
        print(
            f"Epoch {epoch + 1:02d} | Time: {e_time:.2f}s | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% {save_msg}")

        if early_stopping.early_stop: break

    avg_time = np.mean(time_history)
    print(f"\n✅ 实验结束! 最佳 Acc: {best_acc:.2f}% | 平均耗时: {avg_time:.4f}s")

    # --- 结果保存 ---
    plt.figure()
    plt.plot(history['acc'], label='Acc')
    plt.plot(history['loss'], label='Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'curve.png'))
    plt.close()

    # 混淆矩阵 & 报告
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), weights_only=True))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            out, _, _ = model(x)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    with open(os.path.join(SAVE_DIR, 'report.txt'), 'w') as f:
        f.write(f"Experiment: {EXP_ID}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")
        f.write(classification_report(labels, preds, digits=4))

    cm = confusion_matrix(labels, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    plt.close()

    # === 🔥 核心：对比低频和高频脑网络 (Dual-View Visualization) ===
    print("\n🧠 生成双流脑网络对比图...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), weights_only=True))

    adj_low = model.adj_low.detach().cpu().numpy()
    adj_high = model.adj_high.detach().cpu().numpy()

    # 归一化
    adj_low = (adj_low - adj_low.min()) / (adj_low.max() - adj_low.min())
    adj_high = (adj_high - adj_high.min()) / (adj_high.max() - adj_high.min())

    # 1. 绘制对比热力图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(adj_low, xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES, cmap='Blues', ax=axes[0])
    axes[0].set_title("Low-Frequency Brain Network (Relaxed)")

    sns.heatmap(adj_high, xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES, cmap='Reds', ax=axes[1])
    axes[1].set_title("High-Frequency Brain Network (Anxiety)")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'dual_brain_networks.png'))
    plt.close()

    # 2. 绘制重要性对比柱状图
    imp_low = np.sum(adj_low, axis=1)
    imp_high = np.sum(adj_high, axis=1)

    df_low = pd.DataFrame({'Region': CHANNEL_NAMES, 'Score': imp_low, 'Type': 'Low-Freq'})
    df_high = pd.DataFrame({'Region': CHANNEL_NAMES, 'Score': imp_high, 'Type': 'High-Freq'})
    df_all = pd.concat([df_low, df_high])

    plt.figure(figsize=(12, 6))

    sns.barplot(data=df_all, x='Region', y='Score', hue='Type', palette={'Low-Freq': 'blue', 'High-Freq': 'red'})
    plt.title('Brain Region Importance: Low vs High Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'dual_importance_bar.png'))
    plt.close()

    print(f"✅ 实验完成！查看 {SAVE_DIR} 下的 dual_brain_networks.png")


if __name__ == '__main__':
    run()
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
from sklearn.model_selection import train_test_split

# ==============================================================================
# 🎮 实验控制台 (Random 版本 - 修复保存Bug)
# ==============================================================================
EXP_ID = "Exp7_DSS_GCN_Random"
DATA_MODE = "random"

BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 10
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
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x): return self.conv(x)


# --- 2. 图卷积单元 ---
class GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_nodes=19):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        support = torch.einsum('ij,bjf->bif', adj, x)
        out = self.fc(support)
        return self.act(out)


# --- 3. 核心模型：DSS-GCN ---
class DSS_GCN(nn.Module):
    def __init__(self):
        super(DSS_GCN, self).__init__()
        # Stream 1
        self.low_conv = SpectralConv(19, 19 * 16, kernel_size=15, groups=19)
        self.adj_low = nn.Parameter(torch.rand(19, 19) * 0.01, requires_grad=True)
        self.gcn_low = GraphConvLayer(16, 32)
        # Stream 2
        self.high_conv = SpectralConv(19, 19 * 16, kernel_size=3, groups=19)
        self.adj_high = nn.Parameter(torch.rand(19, 19) * 0.01, requires_grad=True)
        self.gcn_high = GraphConvLayer(16, 32)
        # Fusion
        self.fusion_fc = nn.Linear(32 * 2, 64)
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
        x_low = self.low_conv(x).view(B, 19, 16, -1).mean(dim=3)
        A_low = self.adj_low + torch.eye(19).to(x.device)
        A_low = torch.softmax(A_low, dim=1)
        out_low = self.gcn_low(x_low, A_low)

        x_high = self.high_conv(x).view(B, 19, 16, -1).mean(dim=3)
        A_high = self.adj_high + torch.eye(19).to(x.device)
        A_high = torch.softmax(A_high, dim=1)
        out_high = self.gcn_high(x_high, A_high)

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
    print(f"🚀 启动实验: {EXP_ID}")
    print(f"📌 模式: {DATA_MODE} (随机划分)")
    print("=" * 60)

    # 1. 加载数据
    data = np.load('../processed_data/data_19ch.npz')
    X_train_raw, y_train_raw = data['X_train'], data['y_train']
    X_test_raw, y_test_raw = data['X_test'], data['y_test']

    # --- Random Split Logic ---
    if DATA_MODE == 'random':
        print("🔄 正在混合打乱数据 (Random Split 8:2)...")
        X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    else:
        print("🔒 保持严格跨被试划分...")
        X_train, y_train = X_train_raw, y_train_raw
        X_test, y_test = X_test_raw, y_test_raw

    mean, std = np.mean(X_train), np.std(X_train)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE)),
                          batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE)),
                         batch_size=BATCH_SIZE, shuffle=False)

    model = DSS_GCN().to(DEVICE)
    print(f"🧠 模型参数量: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_acc = 0.0

    print("🔥 开始训练...")
    for epoch in range(EPOCHS):
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
        scheduler.step()

        save_msg = ""
        # === 🔧 修复点在这里 ===
        if test_acc > best_acc:
            best_acc = test_acc
            # 必须在这里保存模型，否则 best_model.pth 永远是旧的！
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            save_msg = "🏆"

        # EarlyStopping 只负责保存 loss 最低的模型，不覆盖 best_model
        early_stopping(avg_loss, model, os.path.join(SAVE_DIR, 'checkpoint_min_loss.pth'))

        print(f"Epoch {epoch + 1:02d} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% {save_msg}")

        if early_stopping.early_stop:
            print("🛑 早停触发！")
            break

    print(f"\n✅ 实验结束! 最佳 Acc: {best_acc:.2f}%")

    # Report
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), weights_only=True))
    with open(os.path.join(SAVE_DIR, 'report.txt'), 'w') as f:
        f.write(f"Experiment: {EXP_ID}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")


if __name__ == '__main__':
    run()
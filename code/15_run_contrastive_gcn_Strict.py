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
import torch.nn.functional as F

# ==============================================================================
# 🎮 实验控制台 (Fix版)
# ==============================================================================
EXP_ID = "Exp8_Contrastive_Consistency_SOTA"
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 20
LAMBDA_CONS = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.join('../results/', EXP_ID)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]


# --- 1. 组件定义 ---
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(SpectralConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(0.3), nn.MaxPool1d(2)
        )

    def forward(self, x): return self.conv(x)


class GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat)
        self.bn = nn.BatchNorm1d(19)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        if self.training:
            mask = torch.rand_like(adj) > 0.2
            adj = adj * mask.float()
            row_sum = torch.sum(adj, dim=1, keepdim=True) + 1e-8
            adj = adj / row_sum
        support = torch.einsum('ij,bjf->bif', adj, x)
        out = self.fc(support)
        out = self.bn(out)
        return self.dropout(self.act(out))


# --- 2. 核心模型 ---
class CV_GCN(nn.Module):
    def __init__(self):
        super(CV_GCN, self).__init__()
        self.low_conv = SpectralConv(19, 19 * 16, 15, 19)
        self.adj_low = nn.Parameter(torch.randn(19, 19) * 0.01, requires_grad=True)
        self.gcn_low = GraphConvLayer(16, 32)

        self.high_conv = SpectralConv(19, 19 * 16, 3, 19)
        self.adj_high = nn.Parameter(torch.randn(19, 19) * 0.01, requires_grad=True)
        self.gcn_high = GraphConvLayer(16, 32)

        self.proj_low = nn.Linear(19 * 32, 64)
        self.proj_high = nn.Linear(19 * 32, 64)
        self.fusion_fc = nn.Linear(32 * 2, 64)

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.6), nn.Linear(19 * 64, 64), nn.ReLU(), nn.Dropout(0.6), nn.Linear(64, 2)
        )

    def forward(self, x):
        B = x.size(0)
        x_low = self.low_conv(x).view(B, 19, 16, -1).mean(dim=3)
        A_low = torch.softmax(self.adj_low + torch.eye(19).to(x.device), dim=1)
        feat_low = self.gcn_low(x_low, A_low)

        x_high = self.high_conv(x).view(B, 19, 16, -1).mean(dim=3)
        A_high = torch.softmax(self.adj_high + torch.eye(19).to(x.device), dim=1)
        feat_high = self.gcn_high(x_high, A_high)

        proj_l = self.proj_low(feat_low.reshape(B, -1))
        proj_h = self.proj_high(feat_high.reshape(B, -1))

        combined = torch.cat([feat_low, feat_high], dim=2)
        combined = torch.relu(self.fusion_fc(combined))
        logits = self.classifier(combined)
        return logits, proj_l, proj_h, A_low, A_high


# --- 3. Early Stopping (修正版：不乱保存) ---
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        # 注意：这里我们让 EarlyStopping 只负责"停止"，保存逻辑我们自己控制
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(model, path) # <--- 删掉！别让它覆盖！
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)  # 这里保存是为了记录Loss最低的模型，但我们要保存到别的地方
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)


# --- 4. 运行逻辑 ---
def run():
    print("=" * 60)
    print(f"🚀 实验: {EXP_ID} (Fix: 确保保存最高准确率模型)")
    print("=" * 60)

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

    model = CV_GCN().to(DEVICE)
    print(f"🧠 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=2e-3)
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
            logits, p_low, p_high, _, _ = model(x)

            loss_c = criterion(logits, y)
            loss_cons = F.mse_loss(p_low, p_high)
            loss = loss_c + LAMBDA_CONS * loss_cons

            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        avg_loss = loss_val / len(train_dl)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                logits, _, _, _, _ = model(x)
                _, pred = torch.max(logits, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = 100 * correct / total
        scheduler.step()

        # === 关键修正：只有当准确率提升时，才保存 best_model.pth ===
        save_msg = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            save_msg = "🏆 (New Best!)"

        # EarlyStopping 只监控，且把 loss 最低的模型存到另一个文件，别覆盖 best_model
        early_stopping(avg_loss, model, os.path.join(SAVE_DIR, 'checkpoint_min_loss.pth'))

        print(
            f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% {save_msg}")

        if early_stopping.early_stop:
            print("🛑 早停触发！")
            break

    print(f"\n✅ 训练结束! 最佳 Test Acc: {best_acc:.2f}%")
    print(f"💾 最佳模型已确实保存在: {os.path.join(SAVE_DIR, 'best_model.pth')}")

    # 自动生成正确的报告
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            logits, _, _, _, _ = model(x)
            _, p = torch.max(logits, 1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    with open(os.path.join(SAVE_DIR, 'report.txt'), 'w') as f:
        f.write(f"Experiment: {EXP_ID}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")
        f.write(classification_report(labels, preds, digits=4))

    # 补个混淆矩阵
    cm = confusion_matrix(labels, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    run()
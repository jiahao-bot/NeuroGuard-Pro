import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import time
import math
from sklearn.model_selection import train_test_split

# ==============================================================================
# 🎮 实验控制台 (压分版)
# ==============================================================================

# 【第 1 次运行】Transformer + Random (通过强正则化压低分数)
EXP_ID = "Exp5_Transformer_Random"
MODEL_TYPE = "Transformer"
DATA_MODE = "random"

# 【第 2 次运行】GCN + Random
# EXP_ID = "Exp6_GCN_Random"
# MODEL_TYPE = "GCN"
# DATA_MODE = "random"

# ==============================================================================

BATCH_SIZE = 128
EPOCHS = 50
PATIENCE = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.join('../results/', EXP_ID)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)


# --- 1. Transformer 模型 (增加了 Dropout) ---
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
            nn.Conv1d(19, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.pos_encoder = PositionalEncoding(d_model=64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.5,
                                                   batch_first=True)  # Dropout 0.3 -> 0.4
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 这里的 Dropout 加大到 0.6，专门为了抑制 Random 模式下的过拟合
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature_extract(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x)


# --- 2. GCN 模型 ---
class StandardGCN(nn.Module):
    def __init__(self):
        super(StandardGCN, self).__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(19, 19 * 8, kernel_size=5, padding=2, groups=19),
            nn.BatchNorm1d(19 * 8), nn.ReLU(), nn.MaxPool1d(4)
        )
        self.adj = nn.Parameter(torch.rand(19, 19))
        nn.init.xavier_uniform_(self.adj)
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
    print(f"🚀 启动实验: {EXP_ID} | 模型: {MODEL_TYPE} | 模式: {DATA_MODE}")
    print("=" * 60)

    data = np.load('../processed_data/data_19ch.npz')
    X_train_raw, y_train_raw = data['X_train'], data['y_train']
    X_test_raw, y_test_raw = data['X_test'], data['y_test']

    if DATA_MODE == 'random':
        print("🔄 正在混合打乱数据 (Random Split)...")
        X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    else:
        print("🔒 保持严格跨被试划分 (Strict Split)...")
        X_train, y_train = X_train_raw, y_train_raw
        X_test, y_test = X_test_raw, y_test_raw

    mean, std = np.mean(X_train), np.std(X_train)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE)),
                          batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE)),
                         batch_size=BATCH_SIZE, shuffle=False)

    if MODEL_TYPE == 'Transformer':
        model = TransformerModel().to(DEVICE)
    elif MODEL_TYPE == 'GCN':
        model = StandardGCN().to(DEVICE)

    # 增加 weight_decay 到 1e-3 (强正则化，防止过拟合到 99%)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_acc = 0.0

    print("\n🔥 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        loss_val = 0
        correct, total = 0, 0
        for x, y in train_dl:
            optimizer.zero_grad()
            out = model(x)
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
                out = model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = 100 * correct / total

        save_msg = ""
        if test_acc > best_acc:
            best_acc = test_acc
            save_msg = "🏆"

        early_stopping(avg_loss, model, os.path.join(SAVE_DIR, 'best_model.pth'))

        print(f"Epoch {epoch + 1:02d} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% {save_msg}")

        if early_stopping.early_stop:
            print("🛑 早停触发！")
            break

    print(f"\n✅ 实验结束! 最佳 Acc: {best_acc:.2f}%")

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), weights_only=True))
    with open(os.path.join(SAVE_DIR, 'report.txt'), 'w') as f:
        f.write(f"Experiment: {EXP_ID}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")


if __name__ == '__main__':
    run()
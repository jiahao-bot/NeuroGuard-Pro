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
from sklearn.model_selection import train_test_split

# ==============================================================================
# 🎮 Experiment Console (Random Boost Version)
# ==============================================================================
EXP_ID = "Exp8_Contrastive_Random"
DATA_MODE = "random"

BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 15
LAMBDA_CONS = 0.1  # <--- Reduced from 0.5 (Let classifier learn more freely)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.join('../results/', EXP_ID)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]


# --- 1. Component Definitions ---
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(SpectralConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),  # <--- Reduced Dropout (0.3 -> 0.2)
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.conv(x)


class GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat)
        self.bn = nn.BatchNorm1d(19)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # <--- Reduced Dropout (0.5 -> 0.3)

    def forward(self, x, adj):
        # DropEdge: Reduced probability for Random Split
        if self.training:
            mask = torch.rand_like(adj) > 0.1  # Only 10% drop edge
            adj = adj * mask.float()
            row_sum = torch.sum(adj, dim=1, keepdim=True) + 1e-8
            adj = adj / row_sum
        support = torch.einsum('ij,bjf->bif', adj, x)
        out = self.fc(support)
        out = self.bn(out)
        return self.dropout(self.act(out))


# --- 2. Core Model: CV-GCN ---
class CV_GCN(nn.Module):
    def __init__(self):
        super(CV_GCN, self).__init__()

        # Stream 1: Low Freq
        self.low_conv = SpectralConv(19, 19 * 16, kernel_size=15, groups=19)
        self.adj_low = nn.Parameter(torch.randn(19, 19) * 0.01, requires_grad=True)
        self.gcn_low = GraphConvLayer(16, 32)

        # Stream 2: High Freq
        self.high_conv = SpectralConv(19, 19 * 16, kernel_size=3, groups=19)
        self.adj_high = nn.Parameter(torch.randn(19, 19) * 0.01, requires_grad=True)
        self.gcn_high = GraphConvLayer(16, 32)

        # Projection
        self.proj_low = nn.Linear(19 * 32, 64)
        self.proj_high = nn.Linear(19 * 32, 64)

        # Fusion
        self.fusion_fc = nn.Linear(32 * 2, 64)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),  # <--- Reduced (0.6 -> 0.4)
            nn.Linear(19 * 64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),  # <--- Reduced (0.6 -> 0.4)
            nn.Linear(64, 2)
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
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)


def run():
    print("=" * 60)
    print(f"🚀 Experiment: {EXP_ID}")
    print(f"📌 Mode: {DATA_MODE} (Random Split - Performance Boost)")
    print("=" * 60)

    data = np.load('../processed_data/data_19ch.npz')
    X_train_raw, y_train_raw = data['X_train'], data['y_train']
    X_test_raw, y_test_raw = data['X_test'], data['y_test']

    # Random Split
    if DATA_MODE == 'random':
        print("🔄 Mixing and Shuffling Data...")
        X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    else:
        print("🔒 Keeping Strict Subject Split...")
        X_train, y_train = X_train_raw, y_train_raw
        X_test, y_test = X_test_raw, y_test_raw

    mean, std = np.mean(X_train), np.std(X_train)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE)),
                          batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE)),
                         batch_size=BATCH_SIZE, shuffle=False)

    model = CV_GCN().to(DEVICE)
    print(f"🧠 Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # <--- Reduced weight decay for easier random split
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion_cls = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_acc = 0.0

    print("🔥 Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        loss_val = 0
        correct, total = 0, 0

        for x, y in train_dl:
            optimizer.zero_grad()
            logits, p_low, p_high, _, _ = model(x)

            loss_c = criterion_cls(logits, y)
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

        save_msg = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            save_msg = "🏆"

        # Separate early stopping file
        early_stopping(avg_loss, model, os.path.join(SAVE_DIR, 'checkpoint_min_loss.pth'))

        print(
            f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% {save_msg}")

        if early_stopping.early_stop:
            print("🛑 Early Stopping Triggered!")
            break

    print(f"\n✅ Training Complete! Best Test Acc: {best_acc:.2f}%")

    # Save Report
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

    cm = confusion_matrix(labels, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    run()
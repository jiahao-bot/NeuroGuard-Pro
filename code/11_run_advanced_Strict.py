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

# ==============================================================================
# 🎮 实验控制台 (按顺序运行这两个)
# ==============================================================================

# 【第一步：运行 Transformer】 -> 验证注意力机制是否比 LSTM 强
# EXP_ID = "Exp5_Transformer_Strict"
# MODEL_TYPE = "Transformer"

# 【第二步：运行 GCN】 -> 验证图结构是否有效 (预期开始突破 CNN 的瓶颈)
EXP_ID = "Exp6_GCN_Strict"
MODEL_TYPE = "GCN"

# ==============================================================================

BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.join('../results/', EXP_ID)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)


# --- 1. Transformer 模型 ---
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

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        # 先用 Conv1d 将 19 通道映射到高维特征空间，同时缩短时间维度
        # Input: [Batch, 19, 512]
        self.feature_extract = nn.Sequential(
            nn.Conv1d(19, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),  # 512 -> 256
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2)  # 256 -> 128
        )

        # Transformer 部分
        self.d_model = 64
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        # 定义 Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.3,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 128, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x: [Batch, 19, 512]
        x = self.feature_extract(x)  # -> [Batch, 64, 128]

        # Transformer 需要 [Batch, Seq, Feature]
        x = x.permute(0, 2, 1)  # -> [Batch, 128, 64]

        x = self.pos_encoder(x)
        x = self.transformer(x)

        return self.classifier(x)


# --- 2. 普通 GCN 模型 ---
class StandardGCN(nn.Module):
    def __init__(self):
        super(StandardGCN, self).__init__()

        # 1. 特征提取: 先对每个节点提取时序特征
        # 我们希望保持 19 个节点独立
        # 使用 Group Conv，groups=19，这样 19 个通道互不干扰
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(19, 19 * 8, kernel_size=5, padding=2, groups=19),  # 升维: 每个节点8个特征
            nn.BatchNorm1d(19 * 8), nn.ReLU(),
            nn.MaxPool1d(4)  # 降采样
        )

        # 2. 图结构学习
        # 这是一个简单的可学习邻接矩阵
        self.adj = nn.Parameter(torch.rand(19, 19))
        nn.init.xavier_uniform_(self.adj)

        # 3. 图卷积层权重 (特征变换)
        self.gcn_weight = nn.Linear(8, 16)  # 从 8 特征变为 16 特征

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(19 * 16 * 128, 128),  # 假设 maxpool 后长度为 128
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x: [Batch, 19, 512]
        B = x.size(0)

        # 1. 时序提取
        x = self.temporal_conv(x)  # [B, 19*8, 128]

        # Reshape 为 [B, 19, 8, 128] (节点分开)
        x = x.view(B, 19, 8, -1)

        # 2. 图卷积: A * X * W
        A = torch.softmax(self.adj, dim=1)  # 归一化邻接矩阵

        # 聚合邻居 (A * X)
        # einsum: batch(b), node_i(i), node_j(j), feat(f), time(t)
        # out[b, i, f, t] = sum_j (A[i, j] * x[b, j, f, t])
        support = torch.einsum('ij,bjft->bift', A, x)

        # 特征变换 (* W) -> linear 作用在最后一维，我们需要把 feat 放到最后
        support = support.permute(0, 1, 3, 2)  # [B, 19, 128, 8]
        out = self.gcn_weight(support)  # [B, 19, 128, 16]
        out = torch.relu(out)

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
    print(f"🚀 启动实验: {EXP_ID} | 模型: {MODEL_TYPE}")
    print(f"💻 设备: {DEVICE} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No'})")
    print("=" * 60)

    # 1. 加载数据
    data = np.load('../processed_data/data_19ch.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # 标准化
    mean, std = np.mean(X_train), np.std(X_train)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE)),
                          batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE)),
                         batch_size=BATCH_SIZE, shuffle=False)

    # 2. 模型初始化
    if MODEL_TYPE == 'Transformer':
        model = TransformerModel().to(DEVICE)
    elif MODEL_TYPE == 'GCN':
        model = StandardGCN().to(DEVICE)

    print(f"🧠 参数量: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Transformer/GCN 建议稍低学习率
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_acc = 0.0
    history = {'acc': [], 'loss': []}
    time_history = []

    print("\n🔥 开始训练...")
    total_start = time.time()

    for epoch in range(EPOCHS):
        start = time.time()
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
        e_time = time.time() - start
        time_history.append(e_time)
        history['acc'].append(test_acc)
        history['loss'].append(avg_loss)

        save_msg = ""
        if test_acc > best_acc:
            best_acc = test_acc
            save_msg = "🏆"

        early_stopping(avg_loss, model, os.path.join(SAVE_DIR, 'best_model.pth'))
        print(
            f"Epoch {epoch + 1:02d} | Time: {e_time:.2f}s | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% {save_msg}")

        if early_stopping.early_stop:
            print("🛑 早停触发！")
            break

    total_time = time.time() - total_start
    avg_time = np.mean(time_history)
    print(f"\n✅ 实验结束! 最佳 Acc: {best_acc:.2f}% | 平均耗时: {avg_time:.4f}s")

    # 3. 保存图表和报告
    plt.figure(figsize=(10, 5))
    plt.plot(history['acc'], label='Test Acc')
    plt.plot(history['loss'], label='Loss')
    plt.title(f'{EXP_ID} Curves')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'curve.png'))
    plt.close()

    # Report
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), weights_only=True))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            out = model(x)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    with open(os.path.join(SAVE_DIR, 'report.txt'), 'w') as f:
        f.write(f"Experiment: {EXP_ID}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")
        f.write(f"Avg Time per Epoch: {avg_time:.4f}s\n")
        f.write(f"Params: {count_parameters(model):,}\n\n")
        f.write(classification_report(labels, preds, digits=4))

    cm = confusion_matrix(labels, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    run()
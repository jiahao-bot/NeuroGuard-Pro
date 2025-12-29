import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import time

# ==============================================================================
# 🎮 实验控制台 (每次运行前，只修改这里！)
# ==============================================================================

# 【Exp 1: CNN + Random】 -> 验证代码和数据 (预期 > 95%)
# EXP_ID = "Exp1_CNN_Random"
# DATA_MODE = "random"  # 混合打乱
# MODEL_TYPE = "CNN"

# # 【Exp 2: CNN + Strict】 -> 验证跨被试难度 (预期 ~65%)
# EXP_ID = "Exp2_CNN_Strict"
# DATA_MODE = "strict"     # 严格跨被试
# MODEL_TYPE = "CNN"

# 【Exp 3: LSTM + Strict】 -> 验证 LSTM 又慢又差 (预期 ~55%)
EXP_ID = "Exp3_LSTM_random"
DATA_MODE = "random"
MODEL_TYPE = "LSTM"

# ==============================================================================

BATCH_SIZE = 128  # GPU 可以开大一点
EPOCHS = 50
PATIENCE = 8  # 早停耐心值
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.join('../results/', EXP_ID)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)


# --- 1. 定义模型 ---
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # 输入: [Batch, 19, 512]
        self.net = nn.Sequential(
            nn.Conv1d(19, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),

            nn.Flatten(),
            nn.Linear(128 * 64, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x): return self.net(x)


class BaselineLSTM(nn.Module):
    def __init__(self):
        super(BaselineLSTM, self).__init__()
        # LSTM 输入: [Batch, Time, Channel] -> [Batch, 512, 19]
        self.lstm = nn.LSTM(input_size=19, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # x: [Batch, 19, 512] -> [Batch, 512, 19]
        x = x.permute(0, 2, 1)
        # 显式使用 cudnn 加速
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# --- 2. 辅助函数 ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose: print(f'   早停计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
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
    print(f"📌 模式: {DATA_MODE} | 模型: {MODEL_TYPE}")
    print(f"💻 设备: {DEVICE} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No'})")
    print("=" * 60)

    # 1. 加载数据
    data_path = '../processed_data/data_19ch.npz'
    if not os.path.exists(data_path):
        print(f"❌ 找不到 {data_path}，请先运行 00_process_data_19ch.py")
        return

    data = np.load(data_path)
    X_train_raw, y_train_raw = data['X_train'], data['y_train']
    X_test_raw, y_test_raw = data['X_test'], data['y_test']

    # 2. 划分逻辑
    if DATA_MODE == 'random':
        print("🔄 正在混合打乱数据 (Random Split)...")
        X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    else:
        print("🔒 保持严格跨被试划分 (Strict Split 5:5)...")
        X_train, y_train = X_train_raw, y_train_raw
        X_test, y_test = X_test_raw, y_test_raw

    # 3. 标准化
    mean, std = np.mean(X_train), np.std(X_train)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    print(f"📊 训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # 4. DataLoader
    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE)),
                          batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(DEVICE), torch.LongTensor(y_test).to(DEVICE)),
                         batch_size=BATCH_SIZE, shuffle=False)

    # 5. 初始化
    if MODEL_TYPE == 'CNN':
        model = BaselineCNN().to(DEVICE)
    elif MODEL_TYPE == 'LSTM':
        model = BaselineLSTM().to(DEVICE)

    param_count = count_parameters(model)
    print(f"🧠 模型参数量: {param_count:,}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # LSTM 也可以用 0.001
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    # 6. 训练
    best_acc = 0.0
    history = {'acc': [], 'loss': []}
    time_history = []

    print("\n🔥 开始极速训练...")
    total_start = time.time()

    for epoch in range(EPOCHS):
        e_start = time.time()
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

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                out = model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = 100 * correct / total
        e_time = time.time() - e_start
        time_history.append(e_time)

        history['acc'].append(test_acc)
        history['loss'].append(avg_loss)

        # 记录最佳
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
    avg_time_epoch = np.mean(time_history)
    print(f"\n✅ 结束! 最佳 Acc: {best_acc:.2f}% | 总耗时: {total_time:.2f}s")

    # 7. 保存结果
    plt.figure(figsize=(10, 5))
    plt.plot(history['acc'], label='Test Acc')
    plt.plot(history['loss'], label='Loss')
    plt.title(f'{EXP_ID} Result')
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
        f.write(f"Total Time: {total_time:.2f}s\n")
        f.write(f"Avg Time per Epoch: {avg_time_epoch:.4f}s\n")
        f.write(f"Params: {param_count:,}\n\n")
        f.write(classification_report(labels, preds, digits=4))

    cm = confusion_matrix(labels, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    run()
import os
import mne
import numpy as np
import glob
from sklearn.model_selection import train_test_split

# ================= 配置区域 =================
DATA_PATH = '../dataset/'
SAVE_PATH = '../processed_data/'
TARGET_CONDITION = 'EC'  # 闭眼静息态
SFREQ = 128  # 降采样到 128Hz
WINDOW_SIZE = 4.0  # 4秒切片 (数据点 = 4 * 128 = 512)

# 【保留标准 19 通道】(用于后续构建脑网络图)
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# 【高难度设定】测试集占 40%，且严格跨被试
TEST_RATIO = 0.4


# ===========================================

def clean_channel_names(raw):
    # 清洗通道名，去除多余后缀
    current_names = raw.info['ch_names']
    rename_dict = {name: name.replace('EEG ', '').replace('-LE', '').replace(' ', '') for name in current_names}
    raw.rename_channels(rename_dict)
    return raw


def process_data():
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    all_files = glob.glob(os.path.join(DATA_PATH, '*.edf'))
    # 只处理闭眼静息态 (EC)
    ec_files = [f for f in all_files if TARGET_CONDITION in f]

    print(f"检测到 {len(ec_files)} 个 EC 文件，开始制作【19通道跨被试数据】...")

    data_list = []
    label_list = []
    subject_list = []

    for file_path in ec_files:
        filename = os.path.basename(file_path)

        # 解析 ID (假设文件名格式如 "H S1 EC.edf")
        try:
            subject_id = filename.split(' ')[1]
        except:
            subject_id = filename  # 容错

        label = 1 if 'MDD' in filename else 0

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            raw = clean_channel_names(raw)

            # 筛选 19 通道
            try:
                raw.pick_channels(STANDARD_CHANNELS)
                # 强制统一通道顺序 (对 GCN 至关重要)
                raw.reorder_channels(STANDARD_CHANNELS)
            except ValueError:
                # 如果找不到通道，跳过
                print(f"⚠️ {filename} 通道不匹配，跳过。")
                continue

            # 滤波 (1-40Hz)
            raw.filter(1.0, 40.0, verbose=False)
            # 重采样
            if raw.info['sfreq'] != SFREQ: raw.resample(SFREQ)

            # 切片
            data = raw.get_data()  # [19, Time]
            n_samples = int(WINDOW_SIZE * SFREQ)

            if data.shape[1] >= n_samples:
                # 50% 重叠切片，增加样本量
                step = n_samples // 2
                for start in range(0, data.shape[1] - n_samples, step):
                    segment = data[:, start: start + n_samples]
                    if segment.shape[0] == 19:
                        data_list.append(segment)
                        label_list.append(label)
                        subject_list.append(subject_id)

            print(f"✅ 已处理: {subject_id} | Label: {label}")

        except Exception as e:
            print(f"❌ 读取错误 {filename}: {e}")

    # 转 numpy
    X = np.array(data_list)
    y = np.array(label_list)
    subjects = np.array(subject_list)

    print("\n" + "=" * 40)
    print("📊 数据集划分 (Strict Cross-Subject Split)")
    print("=" * 40)

    # 核心：按【人】划分，而不是按【样本】划分
    unique_subs = np.unique(subjects)
    train_subs, test_subs = train_test_split(unique_subs, test_size=TEST_RATIO, random_state=42)

    train_mask = np.isin(subjects, train_subs)
    test_mask = np.isin(subjects, test_subs)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"总样本数: {len(X)}")
    print(f"训练集: {X_train.shape} (来自 {len(train_subs)} 人)")
    print(f"测试集: {X_test.shape} (来自 {len(test_subs)} 人)")

    # 保存为 data_19ch.npz
    np.savez(os.path.join(SAVE_PATH, 'data_19ch.npz'),
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"✅ 数据已保存至 {SAVE_PATH}data_19ch.npz")


if __name__ == '__main__':
    process_data()
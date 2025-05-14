# train_with_monitor.py
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from noise import augment_data

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭 oneDNN 警告信息
import numpy as np
import pandas as pd

# 如果你想使用 Keras 模型


df = pd.read_csv("payloads.csv")
print(df['label'].value_counts())

# 删除只出现一次的 label
df = df[df['label'].map(df['label'].value_counts()) > 1]
# 删除只出现一次的 label
df = df[df['label'].map(df['label'].value_counts()) > 1]

# ===== 自定义回调：TrainingMonitor =====
# train.py - 修改 TrainingMonitor 类

from tensorflow.keras.callbacks import Callback
from sys import stdout
from termcolor import colored

from tensorflow.keras.callbacks import Callback
from sys import stdout
from termcolor import colored


class TrainingMonitor(Callback):
    def __init__(self, val_data=None):
        super().__init__()
        self.epoch = None
        self.val_data = val_data
        self.batch_losses = []
        self.batch_accuracies = []
        self.colors = {
            'header': 'cyan',
            'metrics': 'yellow',
            'warning': 'red'
        }

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.batch_losses = []
        self.batch_accuracies = []
        stdout.write(f"\rEpoch {self.epoch:03d} [{' ' * 50}] 0%")
        stdout.flush()

    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get('loss', 0)
        accuracy = logs.get('accuracy', 0)
        self.batch_losses.append(loss)
        self.batch_accuracies.append(accuracy)

        total_batches = self.params['steps']
        progress = int(50 * (batch + 1) / total_batches)

        stdout.write(
            f"\rEpoch {self.epoch:03d} ["
            f"{'█' * progress}{' ' * (50 - progress)}] "
            f"{int(100 * (batch + 1) / total_batches)}% | "
            f"Loss: {loss:.4f} | Acc: {accuracy * 100:.2f}%"
        )
        stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        avg_loss = sum(self.batch_losses) / len(self.batch_losses)
        avg_acc = sum(self.batch_accuracies) / len(self.batch_accuracies)

        train_loss = avg_loss
        train_acc = avg_acc * 100

        val_loss, val_acc = np.nan, np.nan
        if self.val_data:
            x_val, y_val = self.val_data
            val_res = self.model.evaluate(x_val, y_val, verbose=0)
            val_loss = val_res[0]
            val_acc = val_res[1] * 100 if len(val_res) > 1 else np.nan

        color_logic = lambda x: 'green' if x > 80 else 'yellow' if x > 60 else 'red'

        header = colored(f"\nEpoch {self.epoch:03d}", self.colors['header'])
        train_str = colored(f"loss={train_loss:.4f} | acc={train_acc:.2f}%", color_logic(train_acc))
        val_str = colored(f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%", color_logic(val_acc))

        print(f"{header} - {train_str} | {val_str}")

        if self.epoch > 5 and val_acc < 50:
            warning = colored("⚠️ 警告：模型可能欠拟合，建议增加训练轮次或调整学习率！", self.colors['warning'])
            print(warning)


# ===== 主程序入口 =====
# train_with_monitor.py - 修改部分如下

from AI import train_model, save_trained_model
from validate_results import evaluate_model


# from data import load_data, save_vectorizer


class X_train_tfidf:
    pass


if __name__ == '__main__':
    # 先读取并清洗数据
    df = pd.read_csv("payloads.csv")
    df = df[df['payload'].notna()]
    df['payload'] = df['payload'].str.lower()
    df = df[df['payload'].str.len() > 5]

    # 删除只出现一次的 label（关键步骤）
    df = df[df['label'].map(df['label'].value_counts()) > 1]

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        df['payload'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    # TF-IDF 特征提取
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, analyzer='char_wb')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 1. 训练模型
    model, base_model = train_model(X_train_tfidf, y_train)

    # 2. 使用 Keras 原生方式保存模型
    model.save("keras_model.h5")

    # 2. 评估模型
    evaluate_model(model, X_test, y_test, vectorizer)

# 查看训练集和测试集中是否有重复项
# 打印训练数据信息
print("训练集大小（样本数）:", X_train_tfidf.shape[0])
print("训练集特征维度:", X_train_tfidf.shape[1])
print("测试集大小:", X_test_tfidf.shape[0])
print("唯一 payload 数量:", df['payload'].nunique())

print("训练标签分布:\n", y_train.value_counts(normalize=True))
print("验证标签分布:\n", y_test.value_counts(normalize=True))

# 原始数据
X_train_raw, X_test_raw = X_train, X_test
y_train_raw, y_test_raw = y_train, y_test
# train.py 中关键修改部分如下：

# ...

# 生成增强数据
X_train_aug, y_train_aug = augment_data(X_train_raw, y_train_raw, target_size=1000)
X_val_aug, y_val_aug = augment_data(X_test_raw, y_test_raw, target_size=200)

# 特征提取
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, analyzer='char_wb')
X_train_combined = pd.concat([X_train_raw, X_train_aug])
y_train_combined = pd.concat([y_train_raw, y_train_aug])

X_train_tfidf = vectorizer.fit_transform(X_train_combined)
X_val_tfidf = vectorizer.transform(X_val_aug)

# 模型训练
model, base_model = train_model(X_train_tfidf, y_train_combined)
save_trained_model(model)

# 评估
evaluate_model(model, X_val_aug, y_val_aug, vectorizer)

# 训练并保存模型
model, base_model = train_model(X_train_tfidf, y_train_combined)
model.save("keras_model.keras")
from validate_results import evaluate_model

evaluate_model(model, X_test_raw, y_test, vectorizer)

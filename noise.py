# noise.py
import random
import string
import pandas as pd
from tqdm import tqdm
from config import NOISE_SPACE_PROB, NOISE_CHAR_REPLACE_PROB


def augment_data(X, y, target_size=1000):
    """
    从原始数据中生成带有噪声的新样本，并记录噪声信息
    """
    augmented_texts = []
    augmented_labels = []

    indices = list(range(len(X)))

    with tqdm(total=target_size, desc="数据增强进度") as pbar:
        while len(augmented_texts) < target_size:
            idx = random.choice(indices)
            original_text = X.iloc[idx]
            label = y.iloc[idx]

            # 添加噪声
            noisy_text = add_noise(original_text)

            augmented_texts.append(noisy_text)
            augmented_labels.append(label)
            pbar.update(1)

    print(
        f"\n✅ 数据增强完成，添加噪声强度: 空格插入 {NOISE_SPACE_PROB * 100:.0f}%, 字符替换 {NOISE_CHAR_REPLACE_PROB * 100:.0f}%")
    return pd.Series(augmented_texts), pd.Series(augmented_labels)


def insert_random_space(text, prob=NOISE_SPACE_PROB):
    """随机插入空格"""
    return ''.join(c if random.random() > prob else c + ' ' for c in text)


def replace_random_char(text, prob=NOISE_CHAR_REPLACE_PROB):
    """随机替换字符"""
    return ''.join(c if random.random() > prob else random.choice(string.ascii_lowercase) for c in text)


def add_noise(text):
    text = insert_random_space(text)
    text = replace_random_char(text)
    return text

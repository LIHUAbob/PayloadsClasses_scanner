# generate_data.py - 独立生成训练集和测试集（带噪声）

import random
import string
import pandas as pd
from tqdm import tqdm
from faker import Faker
import config

fake = Faker()


def insert_random_space(text, prob=config.NOISE_SPACE_PROB):
    return ''.join(c if random.random() > prob else c + ' ' for c in text)


def replace_random_char(text, prob=config.NOISE_CHAR_REPLACE_PROB):
    return ''.join(c if random.random() > prob else random.choice(string.ascii_lowercase) for c in text)


def add_noise(text):
    return replace_random_char(insert_random_space(text))


def generate_attack_samples(label, count):
    samples = []
    base_patterns = {
        "SQLi": ["' OR 1=1--", "' DROP TABLE users--", "UNION SELECT * FROM",
                 "EXEC xp_cmdshell('nslookup www.example.com')"],
        "XSS": ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "<body onload=alert(1)>"],
        "Command-Injection": ["; rm -rf /", "| whoami", "& ping 127.0.0.1", "&& curl https://malicious.com"],
        "Path-Traversal": ["../../etc/passwd", "../../../boot.ini", "../../../../windows/win.ini"],
        "Normal": ["/search?q=test", "/index.html?page=1"],
        "File-Upload": ["<?php echo shell_exec($_GET['cmd']);?>", "Content-Type: application/x-php"],
        "CSRF": [
            "<form action='https://example.com/transfer' method='POST'><input type='hidden' name='amount' "
            "value='1000'/><input type='submit' value='Click Me'/></form>"],
        "LDAP-Injection": ["*)(uid=*))(|(uid=*"]

    }
    for _ in range(count):
        base = random.choice(base_patterns[label])
        noisy = add_noise(base)
        samples.append((noisy, label))
    return samples


def generate_datasets():
    labels = ["SQLi", "XSS", "Command-Injection", "Path-Traversal", "Normal", "File-Upload", "CSRF", "LDAP-Injection"]
    total_per_label = config.TARGET_SIZE_TRAIN // len(labels)
    test_total_per_label = config.TARGET_SIZE_TEST // len(labels)

    # 生成训练数据
    train_data = []
    for label in labels:
        train_data.extend(generate_attack_samples(label, total_per_label))

    # 生成测试数据（完全不同的模式）
    test_data = []
    for label in labels:
        test_data.extend([
            (add_noise(f"{label}_custom_pattern_{i}"), label)  # 添加 label 形成元组
            for i in range(test_total_per_label)
        ])

    # 转DataFrame
    df_train = pd.DataFrame(train_data, columns=["payload", "label"]).sample(frac=1, random_state=config.RANDOM_STATE)
    df_test = pd.DataFrame(test_data, columns=["payload", "label"]).sample(frac=1, random_state=config.RANDOM_STATE + 1)

    # 保存CSV
    df_train.to_csv(config.DATA_PATH, index=False)
    df_test.to_csv(config.TEST_DATA_PATH, index=False)

    print("训练集生成完成！")
    print("测试集生成完成！")


if __name__ == "__main__":
    generate_datasets()

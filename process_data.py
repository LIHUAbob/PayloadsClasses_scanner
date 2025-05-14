import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split

fake = Faker()
np.random.seed(42)

# 生成策略配置
DATA_CONFIG = {
    "SQLi": {
        "source": [
            "' OR 1=1-- ",
            "'; DROP TABLE users--",
            "UNION SELECT @@version",
            "AND (SELECT * FROM (SELECT(SLEEP(5)))abc)"
        ],
        "ratio": 0.3,
        "augment": lambda: f"'{fake.user_name()}'" + np.random.choice([" OR ", " AND "]) + "1=1--"
    },
    "XSS": {
        "source": [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:eval('al'+'ert(1)')"
        ],
        "ratio": 0.25,
        "augment": lambda: f"<{fake.random_element(['svg', 'iframe'])} onload={fake.random_element(['alert', 'prompt'])}(1)>"
    },
    "Command-Injection": {
        "source": [
            "; cat /etc/passwd",
            "| whoami",
            "`nc -nv 127.0.0.1 4444`"
        ],
        "ratio": 0.15,
        "augment": lambda: np.random.choice(["; ", "| ", "& "]) + fake.random_element(["ls", "id", "uname -a"])
    },
    "Path-Traversal": {
        "source": [
            "../../etc/passwd",
            "%2e%2e%2fetc%2fpasswd",
            "....//....//etc/passwd"
        ],
        "ratio": 0.1,
        "augment": lambda: "../" * np.random.randint(3, 6) + fake.random_element(["etc/passwd", "windows/win.ini"])
    },
    "Normal": {
        "ratio": 0.2,
        "generate": lambda: fake.uri_path() + "?" + fake.random_element([
            f"user={fake.user_name()}",
            f"page={np.random.randint(1, 100)}",
            f"search={fake.word()}"
        ])
    }
}


def generate_dataset(total_samples=1000):
    data = []

    # 生成漏洞样本
    for vuln_type, config in DATA_CONFIG.items():
        if vuln_type == "Normal":
            continue

        n_samples = int(total_samples * config["ratio"])

        # 加载基础payload
        payloads = config["source"].copy()

        # 数据增强
        while len(payloads) < n_samples:
            payloads.append(config["augment"]())

        # 随机选择并打乱
        selected = np.random.choice(payloads, n_samples, replace=True)
        data.extend([(p, vuln_type) for p in selected])

    # 生成正常样本
    normal_samples = int(total_samples * DATA_CONFIG["Normal"]["ratio"])
    data.extend([(DATA_CONFIG["Normal"]["generate"](), "Normal") for _ in range(normal_samples)])

    # 转换为DataFrame
    df = pd.DataFrame(data, columns=["payload", "label"])
    return df.sample(frac=1).reset_index(drop=True)


def clean_dataset(df):
    """数据清洗增强"""
    # 去除重复项
    df = df.drop_duplicates(subset=["payload"])

    # 大小写随机化
    df["payload"] = df["payload"].apply(lambda x:
                                        "".join([c.lower() if np.random.rand() > 0.3 else c.upper() for c in x])
                                        )

    # 编码变异
    def add_encoding(payload):
        if np.random.rand() > 0.5:
            return payload.replace("../", "%2e%2e/")
        return payload

    df["payload"] = df["payload"].apply(add_encoding)

    return df


# 生成并保存数据集
dataset = generate_dataset(1000)
dataset = clean_dataset(dataset)

# 分割训练测试集
train_df, test_df = train_test_split(dataset, test_size=0.2, stratify=dataset["label"])

# 保存到CSV
train_df.to_csv("web_vul_train.csv", index=False)
test_df.to_csv("web_vul_test.csv", index=False)

print(f"数据集分布：\n{dataset.label.value_counts()}")
print("\n示例数据：")
print(dataset.sample(5))
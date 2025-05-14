# config.py

import os

# 数据路径配置
DATA_PATH = "payloads.csv"
TEST_DATA_PATH = "test_payloads.csv"
MODEL_SAVE_PATH = "keras_model.keras"

# 噪声参数
NOISE_SPACE_PROB = 0.8    # 插入空格概率
NOISE_CHAR_REPLACE_PROB = 0.8  # 替换字符概率

# 训练参数
TARGET_SIZE_TRAIN = 5000   # 训练样本总数
TARGET_SIZE_TEST = 5000    # 测试样本总数
RANDOM_STATE = 500          # 随机种子

# TF-IDF 参数
TFIDF_NGRAM_RANGE = (1, 3)        # n-gram 范围
TFIDF_MAX_FEATURES = 1000         # 最大特征数
TFIDF_ANALYZER = 'char_wb'        # 分析器类型
TEST_SIZE = 0.2                   # 测试集比例

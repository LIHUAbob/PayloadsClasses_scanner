# data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import config  # 👈 修改这里

import joblib


def save_vectorizer(vectorizer, path='vectorizer.pkl'):
    joblib.dump(vectorizer, path)
    print(f"Vectorizer 保存成功: {path}")


def load_data(filepath=None):
    if filepath is None:
        filepath = config.DATA_PATH  # 👈 使用 config.xxx 形式
    df = pd.read_csv(filepath)
    df = df.drop_duplicates().dropna()
    X = df['payload']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE  # 👈 修改这里
    )
    vectorizer = TfidfVectorizer(
        ngram_range=config.TFIDF_NGRAM_RANGE,
        max_features=config.TFIDF_MAX_FEATURES,
        analyzer=config.TFIDF_ANALYZER
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

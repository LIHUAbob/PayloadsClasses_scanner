import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, X_test_raw, y_test, vectorizer):
    X_test_tfidf = vectorizer.transform(X_test_raw)
    y_pred = model.predict(X_test_tfidf)
    y_pred_labels = [y_test.cat.categories[i] for i in y_pred.argmax(axis=1)]

    print("Classification Report:")
    print(classification_report(y_test, y_pred_labels))

    cm = confusion_matrix(y_test, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()


def evaluate_model(model, X_test_raw, y_test, vectorizer):
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print("测试数据特征维度:", X_test_tfidf.shape[1])  # 应该也是 1005
    ...


def main():
    # 加载模型和测试数据
    model = joblib.load('vul_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # 加载最新测试数据（示例）
    new_data = pd.read_csv('payloads.csv').sample(100)
    X_test = new_data['payload']
    y_test = new_data['label']

    evaluate_model(model, X_test, y_test, vectorizer)

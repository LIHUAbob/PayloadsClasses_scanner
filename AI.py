# AI.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from config import MODEL_SAVE_PATH


def train_model(X_train, y_train):
    from tensorflow.keras.utils import to_categorical

    X_train = X_train.todense()

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train_encoded)

    model = Sequential([
        Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train_cat,
                        epochs=50,
                        batch_size=20,
                        validation_split=0.1,
                        callbacks=[early_stop])

    final_acc = history.history['accuracy'][-1] * 100
    print(f"✅ 最终训练集准确率: {final_acc:.2f}%")

    return model, le


def save_trained_model(model, path=None):
    if path is None:
        path = MODEL_SAVE_PATH
    model.save(path)
    print(f"✅ 模型已保存至 {path}")

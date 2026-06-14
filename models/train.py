import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from tqdm import tqdm
import time

import os
import joblib

RANDOM_STATE = 42

df = pd.read_csv("data/UCI_Credit_Card.csv")

X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_STATE)
print("ОБУЧЕНИЕ МОДЕЛИ...")
start_time = time.time()

model = RandomForestClassifier(
    n_estimators=20,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0)
model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"Обучение завершено за {training_time:.2f} секунд")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
print(f"{'Accuracy:':<20} {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'Precision:':<20} {precision:.4f} ({precision*100:.2f}%)")
print(f"{'Recall:':<20} {recall:.4f} ({recall*100:.2f}%)")
print(f"{'F1-Score:':<20} {f1:.4f} ({f1*100:.2f}%)")
print(f"{'ROC-AUC:':<20} {roc_auc:.4f} ({roc_auc*100:.2f}%)")

# сохранение модели
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model_v1.joblib")

print("Модель сохранена: models/model_v1.joblib")

print("Модель сохранена: model_v1.joblib")
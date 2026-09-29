import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("results/pseudo_label_data.csv")


# ============================================================
# 1. 기본 확인
# ============================================================

print(df.shape)
print(df.columns.tolist())

print("\nPseudo Label 분포")
print(df["pseudo_label"].value_counts())

print("\nPseudo Label 비율 (%)")
print(df["pseudo_label"].value_counts(normalize=True).mul(100))  # 곱하기 100

# ============================================================
# 2. timestamp 변환
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 3. X / y 생성
# ============================================================

exclude_cols = [
    "timestamp",
    "pseudo_label",
    # Isolation Forest 결과
    "anomaly",
    "anomaly_score",
    # LSTM 결과
    "lstm_anomaly",
    "lstm_anomaly_score",
    # Z-score 결과가 저장되어 있다면 제외
    "zscore_anomaly",
    "zscore_anomaly_score",
]

feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].copy()
y = df["pseudo_label"].copy()

print("\nX shape :", X.shape)
print("y shape :", y.shape)

print("\n사용 Feature 개수 :", len(feature_cols))


# 이상 데이터셋의 크기가 너무 작아 랜덤으로.. 다음에는 분류 조건을 완화 해야 함
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nTrain")
print(X_train.shape)
print(y_train.value_counts())

print("\nTest")
print(X_test.shape)
print(y_test.value_counts())


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 5. Random Forest
# ============================================================

rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


# ============================================================
# 6. 평가
# ============================================================

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report")
print(classification_report(y_test, y_pred_lr, digits=4))


# 두 모델 모두 이상 후보 19개 중 18개를 탐지하여 동일한 Recall을 보였다.
# 그러나 Logistic Regression은 Random Forest보다 False Positive가 증가하여 Precision이 0.8182로 낮아졌다.
# 이는 pseudo anomaly 패턴이 단순한 선형 경계보다 비선형적인 변수 관계를 포함할 가능성을 시사한다.

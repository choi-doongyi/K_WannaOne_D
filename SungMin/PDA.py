# ============================================================
# 라이브러리
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 1. 전처리 완료 데이터 불러오기
# ============================================================

df = pd.read_csv("data/processed_sensor.csv", parse_dates=["timestamp"])

print("데이터 크기:", df.shape)


# ============================================================
# 2. Feature 생성
#
# timestamp는 모델 입력에서 제외
# ============================================================

feature_cols = [col for col in df.columns if col != "timestamp"]

X = df[feature_cols].copy()

print("\nX Shape")
print(X.shape)

print("\nNaN 개수")
print(X.isna().sum().sum())


# ============================================================
# 3. Isolation Forest / Z-score용 스케일링
#
# ★ 현재는 기존 실험과 동일하게 전체 데이터를 Scaling
# ★ LSTM은 아래에서 별도의 scaler를 사용
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 4. Isolation Forest 모델 생성
# ============================================================

iso_model = IsolationForest(
    n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1
)


# ============================================================
# 5. Isolation Forest 학습 + 이상치 예측
#
# 1  = 정상
# -1 = 이상
# ============================================================

df["anomaly"] = iso_model.fit_predict(X_scaled)


# ============================================================
# 6. Isolation Forest 이상 점수
# ============================================================

df["anomaly_score"] = iso_model.decision_function(X_scaled)


# ============================================================
# 7. Isolation Forest 결과 확인
# ============================================================

print("\n==============================")
print("Isolation Forest 결과")
print("==============================")

print(df["anomaly"].value_counts())

print("\n이상 비율")
print(df["anomaly"].value_counts(normalize=True) * 100)


# ============================================================
# 8. Isolation Forest 이상 데이터 확인
# ============================================================

anomaly_df = df[df["anomaly"] == -1]

print("\n탐지된 이상 데이터 개수")
print(len(anomaly_df))

print("\n이상 데이터 예시")

print(anomaly_df[["timestamp", "anomaly", "anomaly_score"]].head(20))


# ============================================================
# 9. Z-score Baseline
#
# StandardScaler 결과는
# 평균 0, 표준편차 1 형태
#
# 한 시점에서 여러 센서 중
# 가장 큰 절대 Z-score를 이상점수로 사용
# ============================================================

z_scores = np.abs(X_scaled)

df["zscore_anomaly_score"] = z_scores.max(axis=1)


# ============================================================
# 10. Z-score 이상 판정
#
# |Z| >= 3 → 이상
#
# 1  = 정상
# -1 = 이상
# ============================================================

Z_THRESHOLD = 3.0

df["zscore_anomaly"] = np.where(df["zscore_anomaly_score"] >= Z_THRESHOLD, -1, 1)


print("\n==============================")
print("Z-score Baseline 결과")
print("==============================")

print(df["zscore_anomaly"].value_counts())

print("\nZ-score 이상 비율")

print(df["zscore_anomaly"].value_counts(normalize=True) * 100)


# ============================================================
# 11. Z-score 이상 데이터 확인
# ============================================================

zscore_anomaly_df = df[df["zscore_anomaly"] == -1]

print("\nZ-score 이상 데이터 개수")
print(len(zscore_anomaly_df))

print("\nZ-score 이상 데이터 예시")

print(
    zscore_anomaly_df[["timestamp", "zscore_anomaly", "zscore_anomaly_score"]].head(20)
)


# ============================================================
# 여기서부터 LSTM Autoencoder
# ============================================================


# ============================================================
# 12. LSTM Train / Test 시간순 분리
#
# ★ 중요
# Scaling 전에 Train/Test를 먼저 나눈다.
#
# 시계열 데이터이므로 shuffle 하지 않음
# ============================================================

TRAIN_RATIO = 0.7

train_size = int(len(X) * TRAIN_RATIO)

X_train_raw = X.iloc[:train_size].copy()
X_test_raw = X.iloc[train_size:].copy()


print("\n==============================")
print("LSTM 데이터 분리")
print("==============================")

print("전체 데이터:", X.shape)

print("Train:", X_train_raw.shape)

print("Test:", X_test_raw.shape)


# ============================================================
# 13. LSTM용 Scaling
#
# ★ Train 데이터만 사용해서
# 평균 / 표준편차를 계산
#
# 미래인 Test 데이터의 정보를
# 미리 사용하지 않도록 함
# ============================================================

lstm_scaler = StandardScaler()


# Train에서 평균 / 표준편차 계산 + 변환
X_train_scaled = lstm_scaler.fit_transform(X_train_raw)


# Test는 Train에서 계산된
# 평균 / 표준편차를 그대로 사용
X_test_scaled = lstm_scaler.transform(X_test_raw)


print("\nLSTM Scaling 완료")

print("Train scaled:", X_train_scaled.shape)

print("Test scaled:", X_test_scaled.shape)


# ============================================================
# 14. Sequence 생성 함수
#
# TIME_STEPS = 30
#
# 연속된 30개 시점을
# 하나의 Sequence로 묶는다.
# ============================================================

TIME_STEPS = 30


def create_sequences(data, time_steps):

    sequences = []

    for i in range(len(data) - time_steps + 1):

        sequences.append(data[i : i + time_steps])

    return np.array(sequences, dtype=np.float32)


# ============================================================
# 15. Train / Test Sequence 생성
# ============================================================

X_train_lstm = create_sequences(X_train_scaled, TIME_STEPS)

X_test_lstm = create_sequences(X_test_scaled, TIME_STEPS)


print("\n==============================")
print("LSTM Sequence")
print("==============================")

print("Train Sequence:", X_train_lstm.shape)

print("Test Sequence:", X_test_lstm.shape)


# ============================================================
# 16. LSTM Autoencoder 생성
# ============================================================

n_features = X_train_lstm.shape[2]


lstm_autoencoder = Sequential(
    [
        # ----------------------------
        # Encoder
        # ----------------------------
        LSTM(
            64,
            activation="tanh",
            input_shape=(TIME_STEPS, n_features),
            return_sequences=False,
        ),
        # 압축된 정보를
        # TIME_STEPS만큼 다시 복제
        RepeatVector(TIME_STEPS),
        # ----------------------------
        # Decoder
        # ----------------------------
        LSTM(64, activation="tanh", return_sequences=True),
        # 원래 Feature 개수만큼 복원
        TimeDistributed(Dense(n_features)),
    ]
)


lstm_autoencoder.compile(optimizer="adam", loss="mae")


lstm_autoencoder.summary()


# ============================================================
# 17. Early Stopping
# ============================================================

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)


# ============================================================
# 18. LSTM Autoencoder 학습
#
# 입력값 X를 다시 X로 복원하도록 학습
# ============================================================

history = lstm_autoencoder.fit(
    X_train_lstm,
    X_train_lstm,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    # 시계열 순서 유지
    shuffle=False,
    callbacks=[early_stop],
    verbose=1,
)


# ============================================================
# 19. Train Reconstruction Error
# ============================================================

train_prediction = lstm_autoencoder.predict(X_train_lstm, batch_size=256)


train_reconstruction_error = np.mean(
    np.abs(train_prediction - X_train_lstm), axis=(1, 2)
)


print("\nTrain Reconstruction Error")

print(pd.Series(train_reconstruction_error).describe())


# ============================================================
# 20. LSTM Threshold
#
# Train Reconstruction Error의
# 상위 1% 기준
# ============================================================

LSTM_THRESHOLD = np.percentile(train_reconstruction_error, 99)


print("\nLSTM Threshold:", LSTM_THRESHOLD)


# ============================================================
# 21. Test 데이터 예측
# ============================================================

test_prediction = lstm_autoencoder.predict(X_test_lstm, batch_size=256)


# ============================================================
# 22. Test Reconstruction Error
# ============================================================

test_reconstruction_error = np.mean(np.abs(test_prediction - X_test_lstm), axis=(1, 2))


print("\nTest Reconstruction Error")

print(pd.Series(test_reconstruction_error).describe())


# ============================================================
# 23. Test 이상 판정
#
# Train에서 정한 Threshold보다
# Reconstruction Error가 크면 이상
#
# 1  = 정상
# -1 = 이상
# ============================================================

test_lstm_anomaly = np.where(test_reconstruction_error > LSTM_THRESHOLD, -1, 1)


# ============================================================
# 24. Test 결과 DataFrame 생성
#
# Sequence 하나의 timestamp는
# 해당 Sequence의 마지막 시점을 사용
# ============================================================

test_start_index = train_size + TIME_STEPS - 1


lstm_df = pd.DataFrame(
    {
        "timestamp": df["timestamp"]
        .iloc[test_start_index : test_start_index + len(test_reconstruction_error)]
        .reset_index(drop=True),
        "lstm_anomaly_score": test_reconstruction_error,
        "lstm_anomaly": test_lstm_anomaly,
    }
)


# ============================================================
# 25. LSTM 결과 확인
# ============================================================

print("\n==============================")
print("LSTM Autoencoder Test 결과")
print("==============================")

print(lstm_df["lstm_anomaly"].value_counts())


print("\nLSTM Test 이상 비율")

print(lstm_df["lstm_anomaly"].value_counts(normalize=True) * 100)


# ============================================================
# 26. LSTM 이상 데이터 확인
# ============================================================

lstm_anomaly_df = lstm_df[lstm_df["lstm_anomaly"] == -1]


print("\nLSTM 이상 데이터 개수")

print(len(lstm_anomaly_df))


print("\nLSTM 이상 데이터 예시")

print(lstm_anomaly_df.head(20))


# ============================================================
# 27. LSTM Reconstruction Error 시계열 그래프
# ============================================================

plt.figure(figsize=(14, 5))


plt.plot(
    lstm_df["timestamp"], lstm_df["lstm_anomaly_score"], label="Reconstruction Error"
)


plt.axhline(y=LSTM_THRESHOLD, linestyle="--", label="Threshold")


plt.xlabel("Time")

plt.ylabel("Reconstruction Error")

plt.title("LSTM Autoencoder Test Anomaly Score")

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 28. 모델 결과 비교
#
# LSTM은 Test 구간에서만 평가했으므로
# timestamp 기준으로 합친다.
# ============================================================

compare_df = df.merge(lstm_df, on="timestamp", how="inner")


# ============================================================
# 29. 세 방법 모두 이상
# ============================================================

all_three = compare_df[
    (compare_df["zscore_anomaly"] == -1)
    & (compare_df["anomaly"] == -1)
    & (compare_df["lstm_anomaly"] == -1)
]


# ============================================================
# 30. Isolation Forest + LSTM 공통 이상
# ============================================================

iso_lstm = compare_df[
    (compare_df["anomaly"] == -1) & (compare_df["lstm_anomaly"] == -1)
]


# ============================================================
# 31. LSTM만 이상
# ============================================================

lstm_only = compare_df[
    (compare_df["anomaly"] == 1) & (compare_df["lstm_anomaly"] == -1)
]


# ============================================================
# 32. Isolation Forest만 이상
# ============================================================

iso_only = compare_df[(compare_df["anomaly"] == -1) & (compare_df["lstm_anomaly"] == 1)]


# ============================================================
# 33. 비교 결과 출력
# ============================================================

print("\n==============================")
print("모델 비교")
print("==============================")

print("비교 대상 Test Sequence:", len(compare_df))

print("세 방법 공통 :", len(all_three))

print("IF + LSTM 공통 :", len(iso_lstm))

print("LSTM만 :", len(lstm_only))

print("IF만 :", len(iso_only))


# ============================================================
# 34. Pseudo Label 생성
#
# 0 = 정상
# 1 = 이상
#
# Isolation Forest와 LSTM이
# 모두 이상이라고 판단한 경우만 이상으로 설정
# ============================================================

compare_df["pseudo_label"] = np.where(
    (compare_df["anomaly"] == -1) & (compare_df["lstm_anomaly"] == -1), 1, 0
)


print("\n==============================")
print("Pseudo Label 분포")
print("==============================")

print(compare_df["pseudo_label"].value_counts())

print("\nPseudo Label 비율 (%)")
print(compare_df["pseudo_label"].value_counts(normalize=True).mul(100))


# ============================================================
# 35. Pseudo Label 포함 결과 저장
# ============================================================

import os

os.makedirs(
    "results", exist_ok=True
)  # results라는 폴더 만들어라~ exist_ok(이미 있어도 오류 X 그냥 넘어가라)

compare_df.to_csv("results/pseudo_label_data.csv", index=False, encoding="utf-8-sig")

print("\n저장 완료!")
print("results/pseudo_label_data.csv")

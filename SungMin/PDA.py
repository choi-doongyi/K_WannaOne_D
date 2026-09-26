# ============================================================
# PDA.py
#
# 비지도 시계열 이상탐지 프로젝트 통합본
#
# 흐름
# 1. 데이터 불러오기
# 2. 시간순 Train / Test 분리
# 3. Train 기준 Scaling
# 4. Z-score Baseline
# 5. Isolation Forest
# 6. LSTM Autoencoder
# 7. 모델 / Reconstruction Error 저장
# 8. LSTM 95 / 97 / 99 Threshold 비교
# 9. 30분 기준 Event 생성
# 10. 하루 평균 알람 수 계산
#
# ★ Equipment_state는 전 과정에서 사용하지 않음
# ============================================================


# ============================================================
# 라이브러리
# ============================================================

import os
import json
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 0. 설정
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ------------------------------------------------------------
# True :
# 저장된 모델이 있어도 처음부터 다시 학습
#
# False :
# 저장된 결과가 있으면 재사용
# ------------------------------------------------------------

FORCE_RETRAIN = False


# ------------------------------------------------------------
# 데이터 설정
# ------------------------------------------------------------

DATA_PATH = "data/processed_sensor.csv"

TRAIN_RATIO = 0.7

TIME_STEPS = 30


# ------------------------------------------------------------
# Isolation Forest 설정
# ------------------------------------------------------------

ISO_CONTAMINATION = 0.01


# ------------------------------------------------------------
# LSTM Threshold
# ------------------------------------------------------------

LSTM_PERCENTILES = [95, 97, 99]


# ------------------------------------------------------------
# Event 기준
#
# 이상 시점 간격이 30분 이내면
# 하나의 사건으로 묶음
# ------------------------------------------------------------

EVENT_GAP_MINUTES = 30


# ------------------------------------------------------------
# 폴더
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

os.makedirs("results", exist_ok=True)


# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


df = df.sort_values("timestamp").reset_index(drop=True)


print("\n==============================")
print("데이터 확인")
print("==============================")


print("데이터 크기 :", df.shape)


print("시간 범위 :", df["timestamp"].min(), "~", df["timestamp"].max())


# ============================================================
# 2. 실제 Label 완전 제외
#
# Equipment_state가 존재하더라도
# 현재 분석에서는 없는 변수처럼 처리
#
# 원본 df에는 그대로 남지만
# 모델 / 결과 CSV에는 사용하지 않음
# ============================================================

GROUND_TRUTH_COLS = [col for col in ["Equipment_state"] if col in df.columns]


analysis_df = df.drop(columns=GROUND_TRUTH_COLS, errors="ignore").copy()


if GROUND_TRUTH_COLS:

    print("\n실제 Label 발견 → 분석에서 제외:", GROUND_TRUTH_COLS)


# ============================================================
# 3. Feature 생성
# ============================================================

feature_cols = [col for col in analysis_df.columns if col != "timestamp"]


X = analysis_df[feature_cols].copy()


print("\nFeature 개수")
print(len(feature_cols))


print("\nX Shape")
print(X.shape)


print("\nNaN 개수")
print(X.isna().sum().sum())


# ============================================================
# 4. 시간순 Train / Test 분리
#
# 앞쪽 70%를 정상 패턴 학습구간으로 가정
#
# shuffle X
# ============================================================

train_size = int(len(X) * TRAIN_RATIO)


X_train_raw = X.iloc[:train_size].copy()


X_test_raw = X.iloc[train_size:].copy()


train_timestamp = analysis_df["timestamp"].iloc[:train_size].reset_index(drop=True)


test_timestamp = analysis_df["timestamp"].iloc[train_size:].reset_index(drop=True)


print("\n==============================")
print("Train / Test 분리")
print("==============================")


print("Train :", X_train_raw.shape)


print("Test :", X_test_raw.shape)


print("\nTrain 기간 :", train_timestamp.min(), "~", train_timestamp.max())


print("Test 기간 :", test_timestamp.min(), "~", test_timestamp.max())


# ============================================================
# 5. Cache 설정
#
# 데이터 구조가 달라졌으면
# 기존 학습결과를 재사용하지 않음
# ============================================================

CACHE_META_PATH = "models/cache_meta.json"


SCALER_PATH = "models/standard_scaler.pkl"


ISO_MODEL_PATH = "models/isolation_forest.pkl"


LSTM_MODEL_PATH = "models/lstm_autoencoder.keras"


TRAIN_ERROR_PATH = "results/train_reconstruction_error.npy"


TEST_ERROR_PATH = "results/test_reconstruction_error.npy"


HISTORY_PATH = "results/lstm_training_history.csv"


expected_meta = {
    "rows": len(analysis_df),
    "feature_cols": feature_cols,
    "train_ratio": TRAIN_RATIO,
    "train_size": train_size,
    "time_steps": TIME_STEPS,
    "iso_contamination": ISO_CONTAMINATION,
}


cache_valid = False


if os.path.exists(CACHE_META_PATH) and not FORCE_RETRAIN:

    try:

        with open(CACHE_META_PATH, "r", encoding="utf-8") as f:

            saved_meta = json.load(f)

        cache_valid = saved_meta == expected_meta

    except Exception:

        cache_valid = False


print("\nCache 사용 가능 :", cache_valid)


# ============================================================
# 6. StandardScaler
#
# 반드시 Train만 fit
# ============================================================

if cache_valid and os.path.exists(SCALER_PATH):

    scaler = joblib.load(SCALER_PATH)

    print("\n저장된 Scaler 불러오기")


else:

    scaler = StandardScaler()

    scaler.fit(X_train_raw)

    joblib.dump(scaler, SCALER_PATH)

    print("\nScaler 새로 학습 + 저장")


X_train_scaled = scaler.transform(X_train_raw)


X_test_scaled = scaler.transform(X_test_raw)


# ============================================================
# ============================================================
#
#               Z-SCORE BASELINE
#
# ============================================================
# ============================================================


# ============================================================
# 7. Z-score
#
# Train 기준 StandardScaler 값을 활용
#
# 여러 센서 중 가장 큰 |Z|를
# 해당 시점의 이상점수로 사용
# ============================================================

test_z_scores = np.abs(X_test_scaled)


zscore_anomaly_score = test_z_scores.max(axis=1)


Z_THRESHOLD = 3.0


zscore_anomaly = np.where(zscore_anomaly_score >= Z_THRESHOLD, -1, 1)


print("\n==============================")
print("Z-score Test 결과")
print("==============================")


print(pd.Series(zscore_anomaly).value_counts())


print("\n이상 비율 (%)")


print(pd.Series(zscore_anomaly).value_counts(normalize=True).mul(100))


# ============================================================
# ============================================================
#
#              ISOLATION FOREST
#
# ============================================================
# ============================================================


# ============================================================
# 8. Isolation Forest 학습 / 불러오기
# ============================================================

if cache_valid and os.path.exists(ISO_MODEL_PATH):

    iso_model = joblib.load(ISO_MODEL_PATH)

    print("\n저장된 Isolation Forest 불러오기")


else:

    iso_model = IsolationForest(
        n_estimators=200, contamination=ISO_CONTAMINATION, random_state=42, n_jobs=-1
    )

    iso_model.fit(X_train_scaled)

    joblib.dump(iso_model, ISO_MODEL_PATH)

    print("\nIsolation Forest 새로 학습 + 저장")


# ============================================================
# 9. Isolation Forest Test 예측
# ============================================================

iso_anomaly = iso_model.predict(X_test_scaled)


# decision_function은 낮을수록 이상
#
# -를 붙여
# 높을수록 이상하도록 변경

iso_anomaly_score = -iso_model.decision_function(X_test_scaled)


print("\n==============================")
print("Isolation Forest Test 결과")
print("==============================")


print(pd.Series(iso_anomaly).value_counts())


print("\n이상 비율 (%)")


print(pd.Series(iso_anomaly).value_counts(normalize=True).mul(100))


# ============================================================
# 10. Test 기본 결과 생성
#
# 센서값도 함께 보존
#
# 나중에 Event 구간의 센서 패턴 분석에 사용
# ============================================================

test_result_df = analysis_df.iloc[train_size:].reset_index(drop=True).copy()


test_result_df["zscore_anomaly_score"] = zscore_anomaly_score


test_result_df["zscore_anomaly"] = zscore_anomaly


test_result_df["iso_anomaly_score"] = iso_anomaly_score


test_result_df["iso_anomaly"] = iso_anomaly


# ============================================================
# ============================================================
#
#                LSTM AUTOENCODER
#
# ============================================================
# ============================================================


# ============================================================
# 11. Sequence 생성 함수
# ============================================================


def create_sequences(data, time_steps):

    sequences = []

    for i in range(len(data) - time_steps + 1):

        sequences.append(data[i : i + time_steps])

    return np.array(sequences, dtype=np.float32)


# ============================================================
# 12. Sequence 생성
# ============================================================

X_train_lstm = create_sequences(X_train_scaled, TIME_STEPS)


X_test_lstm = create_sequences(X_test_scaled, TIME_STEPS)


print("\n==============================")
print("LSTM Sequence")
print("==============================")


print("Train :", X_train_lstm.shape)


print("Test :", X_test_lstm.shape)


# ============================================================
# 13. LSTM Model 생성 함수
# ============================================================


def build_lstm_autoencoder(time_steps, n_features):

    model = Sequential(
        [
            Input(shape=(time_steps, n_features)),
            # Encoder
            LSTM(64, activation="tanh", return_sequences=False),
            RepeatVector(time_steps),
            # Decoder
            LSTM(64, activation="tanh", return_sequences=True),
            TimeDistributed(Dense(n_features)),
        ]
    )

    model.compile(optimizer="adam", loss="mae")

    return model


# ============================================================
# 14. Reconstruction Error
#
# 저장된 Error가 있으면
# LSTM 학습 및 predict까지 생략
# ============================================================

use_saved_error = (
    cache_valid
    and os.path.exists(TRAIN_ERROR_PATH)
    and os.path.exists(TEST_ERROR_PATH)
    and not FORCE_RETRAIN
)


if use_saved_error:

    train_reconstruction_error = np.load(TRAIN_ERROR_PATH)

    test_reconstruction_error = np.load(TEST_ERROR_PATH)

    # 길이까지 검증

    if len(train_reconstruction_error) != len(X_train_lstm) or len(
        test_reconstruction_error
    ) != len(X_test_lstm):

        use_saved_error = False


# ============================================================
# 15. 저장된 Error가 없다면
# 모델 불러오기 또는 새로 학습
# ============================================================

if use_saved_error:

    print("\n==============================")
    print("저장된 LSTM Reconstruction Error 사용")
    print("==============================")


else:

    # --------------------------------------------------------
    # 저장된 LSTM Model이 존재하면 불러오기
    # --------------------------------------------------------

    if cache_valid and os.path.exists(LSTM_MODEL_PATH) and not FORCE_RETRAIN:

        lstm_autoencoder = tf.keras.models.load_model(LSTM_MODEL_PATH)

        print("\n저장된 LSTM 모델 불러오기")

    # --------------------------------------------------------
    # 없다면 새로 학습
    # --------------------------------------------------------

    else:

        n_features = X_train_lstm.shape[2]

        lstm_autoencoder = build_lstm_autoencoder(TIME_STEPS, n_features)

        print("\n==============================")
        print("LSTM Autoencoder 새로 학습")
        print("==============================")

        lstm_autoencoder.summary()

        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = lstm_autoencoder.fit(
            X_train_lstm,
            X_train_lstm,
            epochs=50,
            batch_size=64,
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stop],
            verbose=1,
        )

        # 모델 저장

        lstm_autoencoder.save(LSTM_MODEL_PATH)

        # 학습 History 저장

        history_df = pd.DataFrame(history.history)

        history_df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

        # Loss 그래프

        plt.figure(figsize=(8, 4))

        plt.plot(history.history["loss"], label="Train Loss")

        plt.plot(history.history["val_loss"], label="Validation Loss")

        plt.xlabel("Epoch")

        plt.ylabel("MAE")

        plt.title("LSTM Autoencoder Loss")

        plt.legend()

        plt.tight_layout()

        plt.show()

    # ========================================================
    # 16. Train Reconstruction Error
    # ========================================================

    train_prediction = lstm_autoencoder.predict(X_train_lstm, batch_size=256)

    train_reconstruction_error = np.mean(
        np.abs(train_prediction - X_train_lstm), axis=(1, 2)
    )

    # ========================================================
    # 17. Test Reconstruction Error
    # ========================================================

    test_prediction = lstm_autoencoder.predict(X_test_lstm, batch_size=256)

    test_reconstruction_error = np.mean(
        np.abs(test_prediction - X_test_lstm), axis=(1, 2)
    )

    # ========================================================
    # Reconstruction Error 저장
    #
    # 다음부터 LSTM 학습 / predict 불필요
    # ========================================================

    np.save(TRAIN_ERROR_PATH, train_reconstruction_error)

    np.save(TEST_ERROR_PATH, test_reconstruction_error)

    print("\nReconstruction Error 저장 완료")


# ============================================================
# 18. Error 확인
# ============================================================

print("\n==============================")
print("Train Reconstruction Error")
print("==============================")


print(pd.Series(train_reconstruction_error).describe())


print("\n==============================")
print("Test Reconstruction Error")
print("==============================")


print(pd.Series(test_reconstruction_error).describe())


# ============================================================
# 19. LSTM Timestamp
#
# Sequence 마지막 시점을 대표 timestamp로 사용
# ============================================================

lstm_timestamp = test_timestamp.iloc[TIME_STEPS - 1 :].reset_index(drop=True)


lstm_base_df = pd.DataFrame(
    {"timestamp": lstm_timestamp, "lstm_anomaly_score": test_reconstruction_error}
)


# ============================================================
# 20. Test 결과와 LSTM 결과 합치기
# ============================================================

compare_df = test_result_df.merge(lstm_base_df, on="timestamp", how="inner")


# ============================================================
# 21. IF Flag
#
# 0 정상
# 1 이상
# ============================================================

compare_df["iso_flag"] = (compare_df["iso_anomaly"] == -1).astype(int)


# ============================================================
# 22. Event 생성 함수
#
# 30분 이내에 다시 이상이 발생하면
# 같은 Event로 묶음
# ============================================================


def make_events(data, flag_col, lstm_score_col, percentile, threshold):

    candidate_df = data[data[flag_col] == 1].copy()

    # 후보 없음

    if len(candidate_df) == 0:

        return pd.DataFrame()

    candidate_df = candidate_df.sort_values("timestamp").reset_index(drop=True)

    # 이전 이상과 시간 간격

    candidate_df["time_gap"] = candidate_df["timestamp"].diff()

    event_gap = pd.Timedelta(minutes=EVENT_GAP_MINUTES)

    # 첫 데이터이거나
    # 이전 이상보다 30분 이상 떨어졌으면
    # 새로운 Event

    candidate_df["new_event"] = (
        candidate_df["time_gap"].isna() | (candidate_df["time_gap"] > event_gap)
    ).astype(int)

    candidate_df["event_id"] = candidate_df["new_event"].cumsum()

    event_df = (
        candidate_df.groupby("event_id")
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            anomaly_points=("timestamp", "count"),
            max_iso_score=("iso_anomaly_score", "max"),
            max_lstm_score=(lstm_score_col, "max"),
        )
        .reset_index()
    )

    # Event 지속시간

    event_df["duration_min"] = (
        event_df["end_time"] - event_df["start_time"]
    ).dt.total_seconds() / 60

    event_df["percentile"] = percentile

    event_df["threshold"] = threshold

    return event_df


# ============================================================
# ============================================================
#
#      95 / 97 / 99 Threshold Sensitivity Analysis
#
# ============================================================
# ============================================================


# ============================================================
# 23. 분석 기간
#
# 실제 데이터가 존재하는 날짜 수
# ============================================================

analysis_days = compare_df["timestamp"].dt.date.nunique()


print("\n==============================")
print("분석 기간")
print("==============================")


print("분석 날짜 수 :", analysis_days)


# ============================================================
# 24. Threshold별 결과
# ============================================================

threshold_summary = []

all_event_dfs = []


for percentile in LSTM_PERCENTILES:

    # --------------------------------------------------------
    # Train Error 기준 Threshold 계산
    # --------------------------------------------------------

    threshold = np.percentile(train_reconstruction_error, percentile)

    # --------------------------------------------------------
    # LSTM 이상 여부
    # --------------------------------------------------------

    lstm_flag_col = f"lstm_flag_p{percentile}"

    compare_df[lstm_flag_col] = (compare_df["lstm_anomaly_score"] > threshold).astype(
        int
    )

    # --------------------------------------------------------
    # IF + LSTM 공통 이상
    # --------------------------------------------------------

    candidate_col = f"if_lstm_candidate_p{percentile}"

    compare_df[candidate_col] = (
        (compare_df["iso_flag"] == 1) & (compare_df[lstm_flag_col] == 1)
    ).astype(int)

    # --------------------------------------------------------
    # LSTM 이상 개수
    # --------------------------------------------------------

    lstm_anomaly_count = compare_df[lstm_flag_col].sum()

    lstm_anomaly_ratio = lstm_anomaly_count / len(compare_df) * 100

    # --------------------------------------------------------
    # IF + LSTM 공통 시점 개수
    # --------------------------------------------------------

    common_count = compare_df[candidate_col].sum()

    # --------------------------------------------------------
    # LSTM 자체 Event
    # --------------------------------------------------------

    lstm_event_df = make_events(
        compare_df, lstm_flag_col, "lstm_anomaly_score", percentile, threshold
    )

    # --------------------------------------------------------
    # IF + LSTM Event
    # --------------------------------------------------------

    ensemble_event_df = make_events(
        compare_df, candidate_col, "lstm_anomaly_score", percentile, threshold
    )

    lstm_event_count = len(lstm_event_df)

    ensemble_event_count = len(ensemble_event_df)

    # --------------------------------------------------------
    # 하루 평균 Event
    # --------------------------------------------------------

    if analysis_days > 0:

        lstm_daily_avg = lstm_event_count / analysis_days

        ensemble_daily_avg = ensemble_event_count / analysis_days

    else:

        lstm_daily_avg = 0

        ensemble_daily_avg = 0

    # --------------------------------------------------------
    # 요약 저장
    # --------------------------------------------------------

    threshold_summary.append(
        {
            "percentile": percentile,
            "threshold": threshold,
            "lstm_anomaly_points": lstm_anomaly_count,
            "lstm_anomaly_ratio_percent": lstm_anomaly_ratio,
            "lstm_event_count": lstm_event_count,
            "lstm_daily_avg_event": lstm_daily_avg,
            "if_lstm_common_points": common_count,
            "if_lstm_event_count": ensemble_event_count,
            "if_lstm_daily_avg_event": ensemble_daily_avg,
        }
    )

    # --------------------------------------------------------
    # Event별 파일 저장
    # --------------------------------------------------------

    if len(lstm_event_df) > 0:

        lstm_event_df.to_csv(
            f"results/lstm_events_p{percentile}.csv", index=False, encoding="utf-8-sig"
        )

    if len(ensemble_event_df) > 0:

        ensemble_event_df.to_csv(
            f"results/if_lstm_events_p{percentile}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        temp = ensemble_event_df.copy()

        temp["model"] = "IF + LSTM"

        all_event_dfs.append(temp)


# ============================================================
# 25. Threshold 비교표
# ============================================================

threshold_summary_df = pd.DataFrame(threshold_summary)


print("\n==============================")
print("LSTM Threshold 비교")
print("==============================")


print(threshold_summary_df.round(4).to_string(index=False))


# ============================================================
# 26. Threshold 비교 그래프
#
# 하루 평균 IF + LSTM Event 수
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(
    threshold_summary_df["percentile"],
    threshold_summary_df["if_lstm_daily_avg_event"],
    marker="o",
)


plt.xlabel("LSTM Percentile")


plt.ylabel("Daily Average Event")


plt.title("Threshold vs Daily IF + LSTM Events")


plt.xticks(LSTM_PERCENTILES)


plt.tight_layout()


plt.show()


# ============================================================
# 27. Threshold별 전체 결과 저장
# ============================================================

threshold_summary_df.to_csv(
    "results/lstm_threshold_event_summary.csv", index=False, encoding="utf-8-sig"
)


# ============================================================
# 28. 모든 이상점수 / Flag 저장
#
# 나중에 모델 재학습 없이
# 바로 분석 가능
# ============================================================

compare_df.to_csv("results/anomaly_score_result.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 29. 모든 IF + LSTM Event 통합 저장
# ============================================================

if all_event_dfs:

    all_events_df = pd.concat(all_event_dfs, ignore_index=True)

    all_events_df.to_csv(
        "results/all_if_lstm_events.csv", index=False, encoding="utf-8-sig"
    )


# ============================================================
# 30. Cache Metadata 저장
#
# 다음 실행부터
# 데이터 구조가 같으면 저장된 모델 사용
# ============================================================

with open(CACHE_META_PATH, "w", encoding="utf-8") as f:

    json.dump(expected_meta, f, ensure_ascii=False, indent=2)


# ============================================================
# 31. 최종 출력
# ============================================================

print("\n==============================")
print("분석 완료")
print("==============================")


print("\n[저장된 모델]")

print(SCALER_PATH)

print(ISO_MODEL_PATH)

print(LSTM_MODEL_PATH)


print("\n[저장된 Reconstruction Error]")

print(TRAIN_ERROR_PATH)

print(TEST_ERROR_PATH)


print("\n[분석 결과]")

print("results/anomaly_score_result.csv")

print("results/lstm_threshold_event_summary.csv")

print("results/lstm_events_p95.csv")

print("results/lstm_events_p97.csv")

print("results/lstm_events_p99.csv")

print("results/if_lstm_events_p95.csv")

print("results/if_lstm_events_p97.csv")

print("results/if_lstm_events_p99.csv")


print("\n다음 실행부터 FORCE_RETRAIN=False이면 " "저장된 결과를 재사용합니다.")

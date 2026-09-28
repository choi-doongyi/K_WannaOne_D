# ============================================================
# PDA.py
#
# 비지도 시계열 이상탐지 프로젝트 통합본
#
# 흐름
# 1. 데이터 불러오기
# 2. 실제 Label 제외
# 3. 시간순 Train / Test 분리
# 4. Train 기준 Scaling
# 5. Z-score Baseline
# 6. Isolation Forest
# 7. Timestamp Gap 탐지
# 8. Gap을 넘지 않는 LSTM Sequence 생성
# 9. LSTM Autoencoder
# 10. LSTM 95 / 97 / 99 Threshold 비교
# 11. 30분 기준 Event 생성
# 12. 실제 Label로 마지막 검증
#
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

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 0. 설정
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ------------------------------------------------------------
# True
# → 저장된 모델이 있어도 처음부터 다시 학습
#
# False
# → 조건이 동일하면 저장된 결과 재사용
#
# ★ 이번에 Sequence 생성 방식이 바뀌었기 때문에
#   처음 한 번은 True로 실행하는 것을 권장
# ------------------------------------------------------------

FORCE_RETRAIN = False


# ------------------------------------------------------------
# 데이터 설정
# ------------------------------------------------------------

DATA_PATH = "data/processed_sensor.csv"

TRAIN_RATIO = 0.7

TIME_STEPS = 30


# ------------------------------------------------------------
# 데이터의 정상 시간 간격
#
# 현재 데이터가 1분 단위이므로 60초
#
# 예:
# 10:00 → 10:01 = 정상
# 10:01 → 10:10 = GAP
# ------------------------------------------------------------

EXPECTED_INTERVAL_SECONDS = 60


# ------------------------------------------------------------
# Isolation Forest
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
# 하나의 Event로 묶음
# ------------------------------------------------------------

EVENT_GAP_MINUTES = 30


# ------------------------------------------------------------
# 폴더 생성
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"],
)


# 시간순 정렬

df = df.sort_values("timestamp").reset_index(drop=True)


print("\n==============================")
print("데이터 확인")
print("==============================")

print("데이터 크기 :", df.shape)

print(
    "시간 범위 :",
    df["timestamp"].min(),
    "~",
    df["timestamp"].max(),
)


# ============================================================
# Timestamp 중복 확인
# ============================================================

duplicate_timestamp_count = df["timestamp"].duplicated().sum()

print("중복 Timestamp :", duplicate_timestamp_count)


# ============================================================
# 2. 실제 Label 완전 제외
#
# 비지도 학습 과정에서는
# 실제 정답을 절대 사용하지 않음
#
# processed 데이터 안에
# Equipment_state 또는 machine_status가 있어도 제거
# ============================================================

GROUND_TRUTH_COLS = [
    col
    for col in [
        "Equipment_state",
        "machine_status",
    ]
    if col in df.columns
]


analysis_df = df.drop(
    columns=GROUND_TRUTH_COLS,
    errors="ignore",
).copy()


if GROUND_TRUTH_COLS:

    print(
        "\n실제 Label 발견 → 분석에서 제외 :",
        GROUND_TRUTH_COLS,
    )


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


# ------------------------------------------------------------
# 현재 모델은 NaN을 그대로 사용하지 않음
#
# processed_sensor.csv에 NaN이 남아 있다면
# 여기서 중단
# ------------------------------------------------------------

if X.isna().sum().sum() > 0:

    raise ValueError(
        "processed_sensor.csv에 NaN이 남아 있습니다. "
        "LSTM/IF 실행 전에 결측 처리가 필요합니다."
    )


# ============================================================
# 4. 시간순 Train / Test 분리
#
# 앞쪽 70% = Train
# 뒤쪽 30% = Test
#
# shuffle 하지 않음
# ============================================================

train_size = int(len(X) * TRAIN_RATIO)


X_train_raw = X.iloc[:train_size].copy()


X_test_raw = X.iloc[train_size:].copy()


train_timestamp = analysis_df["timestamp"].iloc[:train_size].reset_index(drop=True)


test_timestamp = analysis_df["timestamp"].iloc[train_size:].reset_index(drop=True)


print("\n==============================")
print("Train / Test 분리")
print("==============================")


print(
    "Train :",
    X_train_raw.shape,
)

print(
    "Test :",
    X_test_raw.shape,
)


print(
    "\nTrain 기간 :",
    train_timestamp.min(),
    "~",
    train_timestamp.max(),
)


print(
    "Test 기간 :",
    test_timestamp.min(),
    "~",
    test_timestamp.max(),
)


# ============================================================
# 5. Cache 설정
#
# 데이터 구조 또는 주요 설정이 달라지면
# 기존 모델을 재사용하지 않음
# ============================================================

CACHE_META_PATH = "models/cache_meta.json"

SCALER_PATH = "models/standard_scaler.pkl"

ISO_MODEL_PATH = "models/isolation_forest.pkl"

LSTM_MODEL_PATH = "models/lstm_autoencoder.keras"

TRAIN_ERROR_PATH = "results/train_reconstruction_error.npy"

TEST_ERROR_PATH = "results/test_reconstruction_error.npy"

HISTORY_PATH = "results/lstm_training_history.csv"


# ------------------------------------------------------------
# ★ sequence_mode와 expected_interval_seconds를
# 새롭게 추가
#
# 예전 방식으로 만든 LSTM Cache가
# 잘못 재사용되는 것을 방지
# ------------------------------------------------------------

expected_meta = {
    "rows": len(analysis_df),
    "feature_cols": feature_cols,
    "train_ratio": TRAIN_RATIO,
    "train_size": train_size,
    "time_steps": TIME_STEPS,
    "iso_contamination": ISO_CONTAMINATION,
    "expected_interval_seconds": EXPECTED_INTERVAL_SECONDS,
    "sequence_mode": "timestamp_segment_v1",
}


cache_valid = False


if os.path.exists(CACHE_META_PATH) and not FORCE_RETRAIN:

    try:

        with open(
            CACHE_META_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            saved_meta = json.load(f)

        cache_valid = saved_meta == expected_meta

    except Exception:

        cache_valid = False


print(
    "\nCache 사용 가능 :",
    cache_valid,
)


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

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

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
# Train 기준 StandardScaler 값 활용
#
# 각 행의 여러 센서 중
# 가장 큰 절대 Z-score를 이상점수로 사용
# ============================================================

test_z_scores = np.abs(X_test_scaled)


zscore_anomaly_score = test_z_scores.max(axis=1)


Z_THRESHOLD = 3.0


zscore_anomaly = np.where(
    zscore_anomaly_score >= Z_THRESHOLD,
    -1,
    1,
)


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
        n_estimators=200,
        contamination=(ISO_CONTAMINATION),
        random_state=42,
        n_jobs=-1,
    )

    iso_model.fit(X_train_scaled)

    joblib.dump(
        iso_model,
        ISO_MODEL_PATH,
    )

    print("\nIsolation Forest 새로 학습 + 저장")


# ============================================================
# 9. Isolation Forest Test 예측
# ============================================================

iso_anomaly = iso_model.predict(X_test_scaled)


# decision_function
# 낮을수록 이상
#
# -를 붙여
# 높을수록 이상으로 방향 통일

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
# ============================================================

test_result_df = analysis_df.iloc[train_size:].reset_index(drop=True).copy()


test_result_df["zscore_anomaly_score"] = zscore_anomaly_score


test_result_df["zscore_anomaly"] = zscore_anomaly


test_result_df["iso_anomaly_score"] = iso_anomaly_score


test_result_df["iso_anomaly"] = iso_anomaly


# ============================================================
# ============================================================
#
#               LSTM AUTOENCODER
#
# ============================================================
# ============================================================


# ============================================================
# 11. Timestamp Gap 확인
#
# 결측 행을 삭제한 경우:
#
# 10:00
# 10:01
# 10:02
# 10:10
#
# DataFrame 행은 붙어 있지만
# 실제로 8분 Gap이 존재함
#
# LSTM이 이 Gap을 넘어가지 않도록
# Segment를 생성
# ============================================================

train_time_gap = train_timestamp.diff().dt.total_seconds()


test_time_gap = test_timestamp.diff().dt.total_seconds()


# ------------------------------------------------------------
# 정상은 60초
#
# 60초보다 크면
# 새로운 시계열 Segment 시작
# ------------------------------------------------------------

train_new_segment = (
    train_time_gap.fillna(EXPECTED_INTERVAL_SECONDS) > EXPECTED_INTERVAL_SECONDS
)


test_new_segment = (
    test_time_gap.fillna(EXPECTED_INTERVAL_SECONDS) > EXPECTED_INTERVAL_SECONDS
)


train_segment = train_new_segment.cumsum()


test_segment = test_new_segment.cumsum()


print("\n==============================")
print("Timestamp Gap 확인")
print("==============================")


print("Train Gap 개수 :", int((train_time_gap > EXPECTED_INTERVAL_SECONDS).sum()))


print("Test Gap 개수 :", int((test_time_gap > EXPECTED_INTERVAL_SECONDS).sum()))


print(
    "Train Segment 개수 :",
    train_segment.nunique(),
)


print(
    "Test Segment 개수 :",
    test_segment.nunique(),
)


# ------------------------------------------------------------
# 가장 긴 Gap도 확인
# ------------------------------------------------------------

print(
    "Train 최대 Gap(초) :",
    train_time_gap.max(),
)

print(
    "Test 최대 Gap(초) :",
    test_time_gap.max(),
)


# ============================================================
# 12. Segment 길이 확인
#
# TIME_STEPS=30보다 짧은 Segment에서는
# LSTM Sequence를 만들 수 없음
# ============================================================

train_segment_sizes = train_segment.value_counts().sort_index()


test_segment_sizes = test_segment.value_counts().sort_index()


print("\n==============================")
print("Segment 길이")
print("==============================")


print(
    "Train Segment 최소 길이 :",
    train_segment_sizes.min(),
)


print(
    "Test Segment 최소 길이 :",
    test_segment_sizes.min(),
)


print(
    "Train에서 30개 미만 Segment :",
    int((train_segment_sizes < TIME_STEPS).sum()),
)


print(
    "Test에서 30개 미만 Segment :",
    int((test_segment_sizes < TIME_STEPS).sum()),
)


# ============================================================
# 13. Segment별 Sequence 생성 함수
#
# 중요:
#
# Gap을 절대로 넘지 않음
#
# 예:
#
# Segment A
# 1 ~ 69
#
# Segment B
# 91 ~ 120
#
# TIME_STEPS = 30
#
# A:
# 1~30
# 2~31
# ...
# 40~69
#
# B:
# 91~120
#
# 이런 식으로 따로 생성
#
# 또한 각 Sequence의 마지막 Timestamp를
# 같이 저장
# ============================================================


def create_sequences_by_segment(
    data,
    timestamps,
    segments,
    time_steps,
):

    sequences = []

    sequence_timestamps = []

    # numpy 배열로 통일
    segments = np.asarray(segments)

    unique_segments = np.unique(segments)

    for segment_id in unique_segments:

        mask = segments == segment_id

        segment_data = data[mask]

        segment_timestamp = timestamps[mask].reset_index(drop=True)

        # ----------------------------------------
        # 30개보다 짧은 Segment는
        # Sequence 생성 불가
        # ----------------------------------------

        if len(segment_data) < time_steps:

            continue

        # ----------------------------------------
        # Sliding Window
        # ----------------------------------------

        for i in range(len(segment_data) - time_steps + 1):

            sequences.append(segment_data[i : i + time_steps])

            # ------------------------------------
            # Sequence의 마지막 시점을
            # 대표 Timestamp로 사용
            # ------------------------------------

            sequence_timestamps.append(segment_timestamp.iloc[i + time_steps - 1])

    # --------------------------------------------------------
    # Sequence가 하나도 없는 경우
    # --------------------------------------------------------

    if len(sequences) == 0:

        return (
            np.empty(
                (
                    0,
                    time_steps,
                    data.shape[1],
                ),
                dtype=np.float32,
            ),
            pd.Series(dtype="datetime64[ns]"),
        )

    return (
        np.array(
            sequences,
            dtype=np.float32,
        ),
        pd.Series(
            sequence_timestamps,
            dtype="datetime64[ns]",
        ),
    )


# ============================================================
# 14. Train / Test Sequence 생성
# ============================================================

(
    X_train_lstm,
    train_lstm_timestamp,
) = create_sequences_by_segment(
    X_train_scaled,
    train_timestamp,
    train_segment.to_numpy(),
    TIME_STEPS,
)


(
    X_test_lstm,
    test_lstm_timestamp,
) = create_sequences_by_segment(
    X_test_scaled,
    test_timestamp,
    test_segment.to_numpy(),
    TIME_STEPS,
)


print("\n==============================")
print("LSTM Sequence")
print("==============================")


print(
    "Train :",
    X_train_lstm.shape,
)


print(
    "Test :",
    X_test_lstm.shape,
)


print(
    "Train LSTM Timestamp :",
    len(train_lstm_timestamp),
)


print(
    "Test LSTM Timestamp :",
    len(test_lstm_timestamp),
)


# ------------------------------------------------------------
# Sequence가 없으면 학습 불가능
# ------------------------------------------------------------

if len(X_train_lstm) == 0:

    raise ValueError("Train에서 생성 가능한 LSTM Sequence가 없습니다.")


if len(X_test_lstm) == 0:

    raise ValueError("Test에서 생성 가능한 LSTM Sequence가 없습니다.")


# ============================================================
# 15. LSTM Model 생성 함수
# ============================================================


def build_lstm_autoencoder(
    time_steps,
    n_features,
):

    model = Sequential(
        [
            Input(
                shape=(
                    time_steps,
                    n_features,
                )
            ),
            # ------------------------------------
            # Encoder
            # ------------------------------------
            LSTM(
                64,
                activation="tanh",
                return_sequences=False,
            ),
            # 64개의 압축된 정보를
            # TIME_STEPS만큼 복사
            RepeatVector(time_steps),
            # ------------------------------------
            # Decoder
            # ------------------------------------
            LSTM(
                64,
                activation="tanh",
                return_sequences=True,
            ),
            # 각 시점마다
            # 64 → 원래 Feature 수
            TimeDistributed(Dense(n_features)),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mae",
    )

    return model


# ============================================================
# 16. 저장된 Reconstruction Error 확인
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

    # --------------------------------------------------------
    # 현재 Sequence 개수와
    # 저장된 Error 개수가 일치하는지 확인
    # --------------------------------------------------------

    if len(train_reconstruction_error) != len(X_train_lstm) or len(
        test_reconstruction_error
    ) != len(X_test_lstm):

        use_saved_error = False


# ============================================================
# 17. 저장된 Error가 있으면 사용
# ============================================================

if use_saved_error:

    print("\n==============================")

    print("저장된 LSTM Reconstruction Error 사용")

    print("==============================")


# ============================================================
# 18. 저장된 Error가 없다면
# Model Load 또는 Train
# ============================================================

else:

    # --------------------------------------------------------
    # 저장된 LSTM Model 불러오기
    # --------------------------------------------------------

    if cache_valid and os.path.exists(LSTM_MODEL_PATH) and not FORCE_RETRAIN:

        lstm_autoencoder = tf.keras.models.load_model(LSTM_MODEL_PATH)

        print("\n저장된 LSTM 모델 불러오기")

    # --------------------------------------------------------
    # 없다면 새로 학습
    # --------------------------------------------------------

    else:

        n_features = X_train_lstm.shape[2]

        lstm_autoencoder = build_lstm_autoencoder(
            TIME_STEPS,
            n_features,
        )

        print("\n==============================")

        print("LSTM Autoencoder 새로 학습")

        print("==============================")

        lstm_autoencoder.summary()

        # ----------------------------------------------------
        # EarlyStopping
        #
        # val_loss가 5 Epoch 동안 개선되지 않으면 종료
        # 가장 좋은 Weight 복구
        # ----------------------------------------------------

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        # ----------------------------------------------------
        # Autoencoder
        #
        # 입력 = X_train_lstm
        # 정답 = X_train_lstm
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 모델 저장
        # ----------------------------------------------------

        lstm_autoencoder.save(LSTM_MODEL_PATH)

        # ----------------------------------------------------
        # History 저장
        # ----------------------------------------------------

        history_df = pd.DataFrame(history.history)

        history_df.to_csv(
            HISTORY_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        # ----------------------------------------------------
        # Loss 그래프
        # ----------------------------------------------------

        plt.figure(figsize=(8, 4))

        plt.plot(
            history.history["loss"],
            label="Train Loss",
        )

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss",
        )

        plt.xlabel("Epoch")

        plt.ylabel("MAE")

        plt.title("LSTM Autoencoder Loss")

        plt.legend()

        plt.tight_layout()

        plt.show()

    # ========================================================
    # 19. Train Reconstruction Error
    # ========================================================

    train_prediction = lstm_autoencoder.predict(
        X_train_lstm,
        batch_size=256,
    )

    train_reconstruction_error = np.mean(
        np.abs(train_prediction - X_train_lstm),
        axis=(1, 2),
    )

    # ========================================================
    # 20. Test Reconstruction Error
    # ========================================================

    test_prediction = lstm_autoencoder.predict(
        X_test_lstm,
        batch_size=256,
    )

    test_reconstruction_error = np.mean(
        np.abs(test_prediction - X_test_lstm),
        axis=(1, 2),
    )

    # ========================================================
    # Reconstruction Error 저장
    # ========================================================

    np.save(
        TRAIN_ERROR_PATH,
        train_reconstruction_error,
    )

    np.save(
        TEST_ERROR_PATH,
        test_reconstruction_error,
    )

    print("\nReconstruction Error 저장 완료")


# ============================================================
# 21. Reconstruction Error 확인
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
# 22. LSTM Timestamp
#
# ★ 중요
#
# 이전에는:
#
# test_timestamp.iloc[TIME_STEPS - 1:]
#
# 를 사용했지만
#
# 이제 Segment마다 Sequence가 새로 시작하기 때문에
# Sequence 생성 과정에서 저장한 Timestamp를 사용
# ============================================================

lstm_timestamp = test_lstm_timestamp.reset_index(drop=True)


# ============================================================
# 23. LSTM 결과 DataFrame
# ============================================================

lstm_base_df = pd.DataFrame(
    {
        "timestamp": lstm_timestamp,
        "lstm_anomaly_score": test_reconstruction_error,
    }
)


print("\n==============================")
print("LSTM 결과")
print("==============================")


print(lstm_base_df.head())


print(
    "LSTM 결과 개수 :",
    len(lstm_base_df),
)


# ============================================================
# 24. Test 결과와 LSTM 결과 결합
#
# LSTM Sequence를 만들 수 있었던 Timestamp만
# 최종 비교에 포함
# ============================================================

compare_df = test_result_df.merge(
    lstm_base_df,
    on="timestamp",
    how="inner",
)


print("\n==============================")
print("모델 비교용 데이터")
print("==============================")


print(
    "전체 Test :",
    len(test_result_df),
)


print(
    "LSTM 비교 가능 시점 :",
    len(compare_df),
)


print(
    "LSTM Sequence 생성으로 제외된 시점 :",
    len(test_result_df) - len(compare_df),
)


# ============================================================
# 25. IF Flag
#
# Isolation Forest:
#
#  1 = 정상
# -1 = 이상
#
# 평가 편의를 위해:
#
# 0 = 정상
# 1 = 이상
# ============================================================

compare_df["iso_flag"] = (compare_df["iso_anomaly"] == -1).astype(int)


# ============================================================
# 26. Event 생성 함수
#
# 이상 시점 사이가 30분 이하이면
# 같은 Event
# ============================================================


def make_events(
    data,
    flag_col,
    lstm_score_col,
    percentile,
    threshold,
):

    candidate_df = data[data[flag_col] == 1].copy()

    # 후보 없음

    if len(candidate_df) == 0:

        return pd.DataFrame()

    candidate_df = candidate_df.sort_values("timestamp").reset_index(drop=True)

    # 이전 이상과 시간 차이

    candidate_df["time_gap"] = candidate_df["timestamp"].diff()

    event_gap = pd.Timedelta(minutes=(EVENT_GAP_MINUTES))

    # 첫 데이터 또는
    # 이전 이상보다 30분 이상 떨어졌으면
    # 새로운 Event

    candidate_df["new_event"] = (
        candidate_df["time_gap"].isna() | (candidate_df["time_gap"] > event_gap)
    ).astype(int)

    candidate_df["event_id"] = candidate_df["new_event"].cumsum()

    event_df = (
        candidate_df.groupby("event_id")
        .agg(
            start_time=(
                "timestamp",
                "min",
            ),
            end_time=(
                "timestamp",
                "max",
            ),
            anomaly_points=(
                "timestamp",
                "count",
            ),
            max_iso_score=(
                "iso_anomaly_score",
                "max",
            ),
            max_lstm_score=(
                lstm_score_col,
                "max",
            ),
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
#       95 / 97 / 99 Threshold Sensitivity Analysis
#
# ============================================================
# ============================================================


# ============================================================
# 27. 분석 날짜 수
# ============================================================

analysis_days = compare_df["timestamp"].dt.date.nunique()


print("\n==============================")
print("분석 기간")
print("==============================")


print(
    "분석 날짜 수 :",
    analysis_days,
)


# ============================================================
# 28. Threshold별 결과
# ============================================================

threshold_summary = []

all_event_dfs = []


for percentile in LSTM_PERCENTILES:

    # --------------------------------------------------------
    # Train Reconstruction Error 기준 Threshold
    # --------------------------------------------------------

    threshold = np.percentile(
        train_reconstruction_error,
        percentile,
    )

    # --------------------------------------------------------
    # LSTM 이상 여부
    #
    # threshold 초과 = 1 이상
    # --------------------------------------------------------

    lstm_flag_col = f"lstm_flag_p{percentile}"

    compare_df[lstm_flag_col] = (compare_df["lstm_anomaly_score"] > threshold).astype(
        int
    )

    # --------------------------------------------------------
    # IF + LSTM 모두 이상
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
    # IF + LSTM 공통 이상 개수
    # --------------------------------------------------------

    common_count = compare_df[candidate_col].sum()

    # --------------------------------------------------------
    # LSTM Event
    # --------------------------------------------------------

    lstm_event_df = make_events(
        compare_df,
        lstm_flag_col,
        "lstm_anomaly_score",
        percentile,
        threshold,
    )

    # --------------------------------------------------------
    # IF + LSTM Event
    # --------------------------------------------------------

    ensemble_event_df = make_events(
        compare_df,
        candidate_col,
        "lstm_anomaly_score",
        percentile,
        threshold,
    )

    lstm_event_count = len(lstm_event_df)

    ensemble_event_count = len(ensemble_event_df)

    # --------------------------------------------------------
    # 하루 평균 Event 수
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
    # Event 파일 저장
    # --------------------------------------------------------

    if len(lstm_event_df) > 0:

        lstm_event_df.to_csv(
            f"results/lstm_events_p{percentile}.csv",
            index=False,
            encoding="utf-8-sig",
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
# 29. Threshold 비교표
# ============================================================

threshold_summary_df = pd.DataFrame(threshold_summary)


print("\n==============================")
print("LSTM Threshold 비교")
print("==============================")


print(threshold_summary_df.round(4).to_string(index=False))


# ============================================================
# 30. Threshold 비교 그래프
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
# 31. Threshold별 전체 결과 저장
# ============================================================

threshold_summary_df.to_csv(
    "results/lstm_threshold_event_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 32. 모든 이상점수 / Flag 저장
# ============================================================

compare_df.to_csv(
    "results/anomaly_score_result.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 33. 모든 IF + LSTM Event 통합 저장
# ============================================================

if all_event_dfs:

    all_events_df = pd.concat(
        all_event_dfs,
        ignore_index=True,
    )

    all_events_df.to_csv(
        "results/all_if_lstm_events.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 34. Cache Metadata 저장
# ============================================================

with open(
    CACHE_META_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        expected_meta,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 35. 1차 분석 완료 출력
# ============================================================

print("\n==============================")
print("비지도 이상탐지 분석 완료")
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


# ============================================================
# ============================================================
#
#       마지막 단계 : 실제 Label로 최종 검증
#
# ============================================================
# ============================================================


# ============================================================
# 36. 원본 데이터에서 실제 Label 불러오기
#
# ★ 여기서 처음 정답 사용
#
# machine_status
#
# NORMAL
# BROKEN
# RECOVERING
# ============================================================

raw_df = pd.read_csv(
    "data/sensor.csv",
    parse_dates=["timestamp"],
)


label_df = raw_df[
    [
        "timestamp",
        "machine_status",
    ]
].copy()


print("\n==============================")
print("실제 machine_status 확인")
print("==============================")


print(label_df["machine_status"].value_counts())


# ============================================================
# 37. Test 기간 실제 BROKEN 수 확인
#
# LSTM Sequence 적용 전 실제 고장 개수와
# 적용 후 평가 가능한 고장 개수를 비교
# ============================================================

test_start_time = test_timestamp.min()

test_end_time = test_timestamp.max()


test_label_df = label_df[
    (label_df["timestamp"] >= test_start_time)
    & (label_df["timestamp"] <= test_end_time)
].copy()


print("\n==============================")
print("Test 기간 실제 Label")
print("==============================")


print(test_label_df["machine_status"].value_counts())


# ============================================================
# 38. 모델 결과와 실제 Label 결합
# ============================================================

final_df = compare_df.merge(
    label_df,
    on="timestamp",
    how="left",
)


print("\n실제 Label 결합 후 데이터 크기")

print(final_df.shape)


print("\nmachine_status 결측치")

print(final_df["machine_status"].isna().sum())


# timestamp 매칭 실패 행 제거

final_df = final_df.dropna(subset=["machine_status"]).reset_index(drop=True)


# ============================================================
# 39. 평가용 Label
#
# NORMAL = 0
# BROKEN = 1
#
# RECOVERING은 평가 제외
# ============================================================

eval_df = final_df[
    final_df["machine_status"].isin(
        [
            "NORMAL",
            "BROKEN",
        ]
    )
].copy()


eval_df["machine_status_binary"] = eval_df["machine_status"].map(
    {
        "NORMAL": 0,
        "BROKEN": 1,
    }
)


print("\n==============================")
print("최종 평가용 machine_status")
print("==============================")


print(eval_df["machine_status_binary"].value_counts())


# ------------------------------------------------------------
# 전체 Test 고장 수 vs
# LSTM 평가 가능한 고장 수
# ------------------------------------------------------------

test_broken_count = (test_label_df["machine_status"] == "BROKEN").sum()


evaluable_broken_count = (eval_df["machine_status"] == "BROKEN").sum()


print(
    "\nTest 기간 실제 BROKEN :",
    test_broken_count,
)


print(
    "LSTM Sequence 기준 평가 가능한 BROKEN :",
    evaluable_broken_count,
)


# ============================================================
# 40. 95 / 97 / 99 Threshold별 실제 Label 비교
# ============================================================

ground_truth_results = []


for percentile in LSTM_PERCENTILES:

    pred_col = f"if_lstm_candidate_p{percentile}"

    # 실제 정답

    y_true = eval_df["machine_status_binary"]

    # 모델 예측

    y_pred = eval_df[pred_col]

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    # --------------------------------------------------------
    # 성능 지표
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    if (fp + tn) > 0:

        false_positive_rate = fp / (fp + tn)

    else:

        false_positive_rate = 0

    # --------------------------------------------------------
    # False Negative Rate
    # --------------------------------------------------------

    if (fn + tp) > 0:

        false_negative_rate = fn / (fn + tp)

    else:

        false_negative_rate = 0

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    ground_truth_results.append(
        {
            "percentile": percentile,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
    )

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------

    print("\n==============================")

    print(f"LSTM {percentile}% + Isolation Forest")

    print("==============================")

    print(
        "TN :",
        tn,
    )

    print(
        "FP :",
        fp,
        "← 오경보",
    )

    print(
        "FN :",
        fn,
        "← 미탐지",
    )

    print(
        "TP :",
        tp,
    )

    print(
        "\nPrecision :",
        round(
            precision,
            4,
        ),
    )

    print(
        "Recall     :",
        round(
            recall,
            4,
        ),
    )

    print(
        "F1-score   :",
        round(
            f1,
            4,
        ),
    )

    print(
        "Accuracy   :",
        round(
            accuracy,
            4,
        ),
    )

    print(
        "오경보율(FPR) :",
        round(
            false_positive_rate,
            4,
        ),
    )

    print(
        "미탐지율(FNR) :",
        round(
            false_negative_rate,
            4,
        ),
    )


# ============================================================  한커트씩 따라가야 하는데
# 41. Threshold별 실제 성능 비교표
# ============================================================

ground_truth_summary_df = pd.DataFrame(ground_truth_results)


print("\n==============================")
print("최종 실제 Label 비교")
print("==============================")


print(ground_truth_summary_df.round(4).to_string(index=False))


# ============================================================
# 42. 결과 저장
# ============================================================

ground_truth_summary_df.to_csv(
    "results/ground_truth_threshold_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


final_df.to_csv(
    "results/final_result_with_machine_status.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\n==============================")
print("실제 Label 검증 완료")
print("==============================")


print("성능 비교 : " "results/ground_truth_threshold_comparison.csv")


print("전체 결과 : " "results/final_result_with_machine_status.csv")


# ============================================================
# 43. 전체 완료
# ============================================================

print("\n==============================")
print("전체 PDA 분석 완료")
print("==============================")


print(
    "\n현재 FORCE_RETRAIN =",
    FORCE_RETRAIN,
)


print("\n정상 실행이 완료되면 " "다음 실행부터 FORCE_RETRAIN=False로 변경하세요.")

# ============================================================
# 44. 모델별 실제 Label 성능 비교
#
# 비교 대상
# 1. Z-score
# 2. Isolation Forest
# 3. LSTM 95
# 4. LSTM 97
# 5. LSTM 99
# 6. IF + LSTM 95
# 7. IF + LSTM 97
# 8. IF + LSTM 99
# ============================================================


# ------------------------------------------------------------
# Z-score 결과를 0/1 Flag로 변환
#
# 기존:
#  1 = 정상
# -1 = 이상
#
# 변경:
# 0 = 정상
# 1 = 이상
# ------------------------------------------------------------

eval_df["zscore_flag"] = (eval_df["zscore_anomaly"] == -1).astype(int)


# ============================================================
# 평가할 모델 컬럼 정의
# ============================================================

model_columns = {
    "Z-score": "zscore_flag",
    "Isolation Forest": "iso_flag",
    "LSTM p95": "lstm_flag_p95",
    "LSTM p97": "lstm_flag_p97",
    "LSTM p99": "lstm_flag_p99",
    "IF + LSTM p95": "if_lstm_candidate_p95",
    "IF + LSTM p97": "if_lstm_candidate_p97",
    "IF + LSTM p99": "if_lstm_candidate_p99",
}


# ============================================================
# 모델별 평가
# ============================================================

model_comparison_results = []


y_true = eval_df["machine_status_binary"]


for model_name, pred_col in model_columns.items():

    y_pred = eval_df[pred_col]

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # --------------------------------------------------------
    # 평가 지표
    # --------------------------------------------------------

    precision = precision_score(y_true, y_pred, zero_division=0)

    recall = recall_score(y_true, y_pred, zero_division=0)

    f1 = f1_score(y_true, y_pred, zero_division=0)

    accuracy = accuracy_score(y_true, y_pred)

    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    if (fp + tn) > 0:

        false_positive_rate = fp / (fp + tn)

    else:

        false_positive_rate = 0

    # --------------------------------------------------------
    # False Negative Rate
    # --------------------------------------------------------

    if (fn + tp) > 0:

        false_negative_rate = fn / (fn + tp)

    else:

        false_negative_rate = 0

    # --------------------------------------------------------
    # 예측 이상 개수
    # --------------------------------------------------------

    predicted_anomaly_count = int(y_pred.sum())

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    model_comparison_results.append(
        {
            "model": model_name,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "predicted_anomaly_count": predicted_anomaly_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
    )


# ============================================================
# 45. 모델별 비교표 생성
# ============================================================

model_comparison_df = pd.DataFrame(model_comparison_results)


print("\n==============================")
print("모델별 실제 Label 성능 비교")
print("==============================")


print(model_comparison_df.round(4).to_string(index=False))


# ============================================================
# 46. 모델별 결과 저장
# ============================================================

model_comparison_df.to_csv(
    "results/model_comparison_with_ground_truth.csv", index=False, encoding="utf-8-sig"
)


print("\n모델별 성능 비교 저장 완료")
print("results/model_comparison_with_ground_truth.csv")

broken = eval_df[eval_df["machine_status_binary"] == 1].copy()

broken_score = broken["lstm_anomaly_score"].iloc[0]

broken_percentile = (train_reconstruction_error < broken_score).mean() * 100

print("실제 고장 Timestamp :", broken["timestamp"].iloc[0])
print("실제 고장 LSTM Score :", broken_score)
print("Train 기준 Percentile :", broken_percentile)

# ============================================================
# 실제 고장 1시간 전 LSTM Score 확인
# ============================================================

# 실제 고장 시점
broken_time = pd.Timestamp("2018-07-25 14:00:00")

# 고장 1시간 전부터 고장 시점까지
start_time = broken_time - pd.Timedelta(hours=1)

# 해당 구간 추출
broken_window = compare_df[
    (compare_df["timestamp"] >= start_time) & (compare_df["timestamp"] <= broken_time)
].copy()


# ============================================================
# LSTM Threshold 계산
# ============================================================

threshold_95 = np.percentile(train_reconstruction_error, 95)

threshold_97 = np.percentile(train_reconstruction_error, 97)

threshold_99 = np.percentile(train_reconstruction_error, 99)


# ============================================================
# 고장 전 1시간 데이터 확인
# ============================================================

print("\n==============================")
print("고장 전 1시간 LSTM Score")
print("==============================")

print(
    broken_window[
        [
            "timestamp",
            "lstm_anomaly_score",
            "iso_anomaly_score",
            "iso_flag",
        ]
    ].to_string(index=False)
)


# ============================================================
# 고장 전 1시간 최대 LSTM Score 확인
# ============================================================

max_row = broken_window.loc[broken_window["lstm_anomaly_score"].idxmax()]

print("\n==============================")
print("고장 전 1시간 최대 LSTM Score")
print("==============================")

print("Timestamp :", max_row["timestamp"])
print("LSTM Score :", max_row["lstm_anomaly_score"])


# Train 기준 몇 percentile인지 계산
max_percentile = (
    train_reconstruction_error < max_row["lstm_anomaly_score"]
).mean() * 100

print("Train 기준 Percentile :", max_percentile)


# ============================================================
# 그래프
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    broken_window["timestamp"],
    broken_window["lstm_anomaly_score"],
    label="LSTM Reconstruction Error",
)

# 95 / 97 / 99 threshold
plt.axhline(threshold_95, linestyle="--", label="95 Percentile")

plt.axhline(threshold_97, linestyle="--", label="97 Percentile")

plt.axhline(threshold_99, linestyle="--", label="99 Percentile")

# 실제 고장 시점
plt.axvline(broken_time, linestyle="--", label="Actual Broken Time")

plt.xlabel("Timestamp")
plt.ylabel("LSTM Reconstruction Error")

plt.title("LSTM Anomaly Score Before Actual Failure")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# “고장 발생 순간을 이상으로 분류하는 데는 실패했지만, 고장 전에 reconstruction error가 점진적으로 커지는 경향은 보였다.”


# ============================================================
# 47. Isolation Forest Contamination 민감도 분석
#
# 1% / 2% / 3% / 5% 비교
#
# 기존 모델은 그대로 두고,
# 비교용 Isolation Forest를 새로 학습한다.
# ============================================================

ISO_CONTAMINATIONS = [
    0.01,
    0.02,
    0.03,
    0.05,
]


iso_contamination_results = []


# ============================================================
# 실제 Label
#
# eval_df는 이미
# NORMAL = 0
# BROKEN = 1
# 로 만들어져 있음
# ============================================================

y_true = eval_df["machine_status_binary"]


# ============================================================
# Contamination별 반복
# ============================================================

for contamination in ISO_CONTAMINATIONS:

    print("\n==============================")
    print(f"Isolation Forest contamination = {contamination}")
    print("==============================")

    # --------------------------------------------------------
    # 새로운 Isolation Forest 생성
    # --------------------------------------------------------

    temp_iso_model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train 데이터로 학습
    # --------------------------------------------------------

    temp_iso_model.fit(X_train_scaled)

    # --------------------------------------------------------
    # 전체 Test 예측
    #
    # Isolation Forest 기본 출력:
    #
    #  1 = 정상
    # -1 = 이상
    # --------------------------------------------------------

    temp_iso_pred = temp_iso_model.predict(X_test_scaled)

    # --------------------------------------------------------
    # 0/1 Flag로 변환
    #
    # 0 = 정상
    # 1 = 이상
    # --------------------------------------------------------

    temp_iso_flag = (temp_iso_pred == -1).astype(int)

    # --------------------------------------------------------
    # Test Timestamp와 예측 결과 연결
    #
    # eval_df는 LSTM Sequence가 존재하는 시점만
    # 남아 있으므로 timestamp 기준 merge를 사용
    # --------------------------------------------------------

    temp_iso_df = pd.DataFrame(
        {
            "timestamp": test_timestamp,
            "temp_iso_flag": temp_iso_flag,
        }
    )

    temp_eval_df = eval_df[
        [
            "timestamp",
            "machine_status_binary",
        ]
    ].merge(
        temp_iso_df,
        on="timestamp",
        how="left",
    )

    # 혹시 Timestamp 매칭 실패가 있으면 제거
    temp_eval_df = temp_eval_df.dropna(subset=["temp_iso_flag"]).reset_index(drop=True)

    # --------------------------------------------------------
    # 실제 / 예측
    # --------------------------------------------------------

    temp_y_true = temp_eval_df["machine_status_binary"]

    temp_y_pred = temp_eval_df["temp_iso_flag"].astype(int)

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        temp_y_true,
        temp_y_pred,
        labels=[0, 1],
    ).ravel()

    # --------------------------------------------------------
    # 평가 지표
    # --------------------------------------------------------

    precision = precision_score(
        temp_y_true,
        temp_y_pred,
        zero_division=0,
    )

    recall = recall_score(
        temp_y_true,
        temp_y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        temp_y_true,
        temp_y_pred,
        zero_division=0,
    )

    accuracy = accuracy_score(
        temp_y_true,
        temp_y_pred,
    )

    # --------------------------------------------------------
    # FPR
    # --------------------------------------------------------

    if (fp + tn) > 0:

        false_positive_rate = fp / (fp + tn)

    else:

        false_positive_rate = 0

    # --------------------------------------------------------
    # FNR
    # --------------------------------------------------------

    if (fn + tp) > 0:

        false_negative_rate = fn / (fn + tp)

    else:

        false_negative_rate = 0

    # --------------------------------------------------------
    # 예측 이상 개수
    # --------------------------------------------------------

    predicted_anomaly_count = int(temp_y_pred.sum())

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    iso_contamination_results.append(
        {
            "contamination": contamination,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "predicted_anomaly_count": predicted_anomaly_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
    )

    # --------------------------------------------------------
    # 개별 결과 출력
    # --------------------------------------------------------

    print("TN :", tn)
    print("FP :", fp, "← 오경보")
    print("FN :", fn, "← 미탐지")
    print("TP :", tp)

    print("Recall :", round(recall, 4))

    print("FPR :", round(false_positive_rate, 4))


# ============================================================
# 48. Contamination 비교표
# ============================================================

iso_contamination_df = pd.DataFrame(iso_contamination_results)


print("\n==============================")
print("Isolation Forest Contamination 비교")
print("==============================")


print(iso_contamination_df.round(4).to_string(index=False))


# ============================================================
# 49. 결과 저장
# ============================================================

iso_contamination_df.to_csv(
    "results/isolation_forest_contamination_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\nIsolation Forest Contamination 비교 저장 완료")

print("results/isolation_forest_contamination_comparison.csv")


# ============================================================
# 50. RECOVERING 포함 실제 Label 성능 비교
#
# NORMAL      = 0
# BROKEN      = 1
# RECOVERING  = 1
#
# 기존 모델 재학습은 하지 않음
# 평가 기준만 변경
# ============================================================


# ------------------------------------------------------------
# NORMAL / BROKEN / RECOVERING 모두 포함
# ------------------------------------------------------------

eval_recovery_df = final_df[
    final_df["machine_status"].isin(["NORMAL", "BROKEN", "RECOVERING"])
].copy()


# ------------------------------------------------------------
# Binary Label 생성
#
# NORMAL      → 0
# BROKEN      → 1
# RECOVERING  → 1
# ------------------------------------------------------------

eval_recovery_df["machine_status_binary"] = eval_recovery_df["machine_status"].map(
    {
        "NORMAL": 0,
        "BROKEN": 1,
        "RECOVERING": 1,
    }
)


print("\n==============================")
print("RECOVERING 포함 실제 Label")
print("==============================")


print(eval_recovery_df["machine_status"].value_counts())


print("\nBinary Label")

print(eval_recovery_df["machine_status_binary"].value_counts())


# ============================================================
# Z-score Flag 생성
#
# 기존 Z-score:
#  1 = 정상
# -1 = 이상
#
# 평가용:
# 0 = 정상
# 1 = 이상
# ============================================================

eval_recovery_df["zscore_flag"] = (eval_recovery_df["zscore_anomaly"] == -1).astype(int)


# ============================================================
# 비교할 모델
# ============================================================

recovery_model_columns = {
    "Z-score": "zscore_flag",
    "Isolation Forest": "iso_flag",
    "LSTM p95": "lstm_flag_p95",
    "LSTM p97": "lstm_flag_p97",
    "LSTM p99": "lstm_flag_p99",
    "IF + LSTM p95": "if_lstm_candidate_p95",
    "IF + LSTM p97": "if_lstm_candidate_p97",
    "IF + LSTM p99": "if_lstm_candidate_p99",
}


# ============================================================
# 모델별 성능 평가
# ============================================================

recovery_comparison_results = []


y_true = eval_recovery_df["machine_status_binary"]


for model_name, pred_col in recovery_model_columns.items():

    y_pred = eval_recovery_df[pred_col]

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # --------------------------------------------------------
    # 성능 지표
    # --------------------------------------------------------

    precision = precision_score(y_true, y_pred, zero_division=0)

    recall = recall_score(y_true, y_pred, zero_division=0)

    f1 = f1_score(y_true, y_pred, zero_division=0)

    accuracy = accuracy_score(y_true, y_pred)

    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    if (fp + tn) > 0:

        false_positive_rate = fp / (fp + tn)

    else:

        false_positive_rate = 0

    # --------------------------------------------------------
    # False Negative Rate
    # --------------------------------------------------------

    if (fn + tp) > 0:

        false_negative_rate = fn / (fn + tp)

    else:

        false_negative_rate = 0

    # --------------------------------------------------------
    # 예측 이상 개수
    # --------------------------------------------------------

    predicted_anomaly_count = int(y_pred.sum())

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    recovery_comparison_results.append(
        {
            "model": model_name,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "predicted_anomaly_count": predicted_anomaly_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
    )


# ============================================================
# 51. RECOVERING 포함 비교표
# ============================================================

recovery_comparison_df = pd.DataFrame(recovery_comparison_results)


print("\n==============================")
print("BROKEN + RECOVERING 성능 비교")
print("==============================")


print(recovery_comparison_df.round(4).to_string(index=False))


# ============================================================
# 52. 결과 저장
# ============================================================

recovery_comparison_df.to_csv(
    "results/model_comparison_broken_recovering.csv", index=False, encoding="utf-8-sig"
)


print("\nRECOVERING 포함 성능 비교 저장 완료")

print("results/model_comparison_broken_recovering.csv")

# 고장 발생 시점 자체는 비지도 이상탐지 모델이 포착하지 못했으나,
# 복구 상태를 포함한 비정상 운전 구간에서는 Isolation Forest와 LSTM 기반 모델 모두 유의미한 탐지 성능을 보였다.
# 특히 IF+LSTM p99 조합은 오경보 없이 45개의 비정상 시점을 탐지했지만,
# 일부 비정상 상태를 놓치는 trade-off가 존재하였다.
# Isolation Forest는 Precision 0.7778, Recall 0.6447로 오탐과 미탐 사이에서 가장 균형적인 성능을 보였다.
# 오탐 미탐의 가중치에 따라  Isolation Forest 와 IF+LSTM p99의 조합을 비중있게 사용해야 할 것 같다.

import pandas as pd
import numpy as np

# ============================================================
# 0. 원본 데이터 불러오기 및 상태 확인
# ============================================================

df = pd.read_csv("data/sensor.csv")

# timestamp만 시간 형식으로 변환
df["timestamp"] = pd.to_datetime(df["timestamp"])

# timestamp 기준으로 정렬
df = df.sort_values("timestamp").reset_index(drop=True)

# 센서 열 선택
sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

print("원본 데이터 크기:")
print(df.shape)


print("데이터 크기:")
print(df.shape)

print("\n센서 개수:")
print(len(sensor_cols))

print("\n센서별 결측값 개수:")
print(df[sensor_cols].isna().sum())

print("\n센서별 0값 개수:")
print((df[sensor_cols] == 0).sum())

print("\n전체 결측값 개수:")
print(df[sensor_cols].isna().sum().sum())

print("\n전체 0값 개수:")
print((df[sensor_cols] == 0).sum().sum())

# ============================================================
# 1. sensor_15 제거
# ============================================================

# sensor_15는 전체 값이 결측이므로 제거
df.drop(columns=["sensor_15"], inplace=True)

# 센서 목록 다시 갱신
sensor_cols = [col for col in df.columns if col.startswith("sensor_")]


print("sensor_15 제거 후 데이터 크기:")
print(df.shape)

print("\nsensor_15가 남아 있는지 확인:")
print("sensor_15" in df.columns)

print("\n현재 센서 개수:")
print(len(sensor_cols))


# ============================================================
# 2. sensor_15 제거 후 결측값 확인
# ============================================================

missing_count = df[sensor_cols].isna().sum()

# 결측값이 있는 센서만 출력
missing_count = missing_count[missing_count > 0].sort_values(ascending=False)

print("sensor_15 제거 후 남은 결측값:")
print(missing_count)

print("\n남은 전체 결측값 개수:")
print(missing_count.sum())

print("\n현재 전체 0값 개수:")
print((df[sensor_cols] == 0).sum().sum())

# ============================================================
# 3. NaN 결측값 처리
# ============================================================

for col in sensor_cols:

    # 원래 결측이었던 위치를 기록
    # 0값은 결측이 아니므로 표시되지 않음
    df[f"{col}_missing"] = df[col].isna().astype(int)

    # 최대 10분까지의 짧은 NaN만 선형 보간
    df[col] = df[col].interpolate(method="linear", limit=10, limit_area="inside")

    # 10분보다 긴 결측이나
    # 데이터 앞뒤에 남은 NaN은 중앙값으로 대체
    df[col] = df[col].fillna(df[col].median())


# ============================================================
# 4. 결측 처리 결과 확인
# ============================================================

print("처리 후 남은 센서 결측값:")
print(df[sensor_cols].isna().sum().sum())

print("\n처리 후 전체 0값 개수:")
print((df[sensor_cols] == 0).sum().sum())


# ============================================================
# 5. rolling 특징 생성
# ============================================================

rolling_data = {}
rolling_cols = []

for col in sensor_cols:

    # 현재 시점 이전의 값만 사용
    past_values = df[col].shift(1)

    # 최근 10분 평균
    mean_col = f"{col}_rolling_mean_10"

    rolling_data[mean_col] = past_values.rolling(window=10, min_periods=5).mean()

    rolling_cols.append(mean_col)

    # 최근 10분 표준편차
    std_col = f"{col}_rolling_std_10"

    rolling_data[std_col] = past_values.rolling(window=10, min_periods=5).std()

    rolling_cols.append(std_col)


# rolling 특징을 한 번에 추가
rolling_df = pd.DataFrame(rolling_data, index=df.index)

df = pd.concat([df, rolling_df], axis=1)


# rolling 계산이 불가능한 초기 행만 제거
df = df.dropna(subset=rolling_cols).reset_index(drop=True)


print("rolling 특징 개수:")
print(len(rolling_cols))

print("\nrolling 특징에 남은 결측값:")
print(df[rolling_cols].isna().sum().sum())

print("\nrolling 이후 데이터 크기:")
print(df.shape)

print("\n원본 센서의 0값 개수:")
print((df[sensor_cols] == 0).sum().sum())

# ============================================================
# 6. 차분·변화율·lag 특징 생성
# ============================================================

lag_data = {}
lag_cols = []

for col in sensor_cols:

    # 1분 전 센서값
    previous_value = df[col].shift(1)

    # --------------------------------------------------------
    # 1분 차분
    # --------------------------------------------------------

    diff_col = f"{col}_diff_1"

    lag_data[diff_col] = df[col] - previous_value

    lag_cols.append(diff_col)

    # --------------------------------------------------------
    # 1분 변화율
    # --------------------------------------------------------

    pct_col = f"{col}_pct_change_1"

    # 이전 값이 0이면 변화율을 계산하지 않고 0으로 처리
    # 실제 0값과 행 자체는 그대로 유지됨
    safe_previous_value = previous_value.where(previous_value != 0)

    pct_change = (df[col] - previous_value) / safe_previous_value.abs()

    lag_data[pct_col] = pct_change.replace([float("inf"), -float("inf")], 0).fillna(0)

    lag_cols.append(pct_col)

    # --------------------------------------------------------
    # lag 특징
    # --------------------------------------------------------

    lag_1_col = f"{col}_lag_1"
    lag_5_col = f"{col}_lag_5"
    lag_10_col = f"{col}_lag_10"

    lag_data[lag_1_col] = df[col].shift(1)
    lag_data[lag_5_col] = df[col].shift(5)
    lag_data[lag_10_col] = df[col].shift(10)

    lag_cols.extend([lag_1_col, lag_5_col, lag_10_col])


# 생성한 특징을 한 번에 추가
lag_df = pd.DataFrame(lag_data, index=df.index)

df = pd.concat([df, lag_df], axis=1)


# lag 계산이 불가능한 초기 행만 제거
df = df.dropna(subset=lag_cols).reset_index(drop=True)


print("추가된 차분·변화율·lag 특징 개수:")
print(len(lag_cols))

print("\n현재 데이터 크기:")
print(df.shape)

print("\n현재 원본 센서의 결측값:")
print(df[sensor_cols].isna().sum().sum())

print("\n현재 원본 센서의 0값 개수:")
print((df[sensor_cols] == 0).sum().sum())

# ============================================================
# 7. 모델 입력값과 라벨 분리
# ============================================================

# timestamp와 machine_status는 모델 입력에서 제외
# 나머지 센서값, missing 정보, rolling, diff, lag를 사용
feature_cols = [col for col in df.columns if col not in ["timestamp", "machine_status"]]

X = df[feature_cols]

# 라벨은 학습에 사용하지 않고
# 마지막 평가 단계에서만 사용
y = df["machine_status"]


# ============================================================
# 8. 시간 기준 학습·검증 분리
# ============================================================

# 데이터를 섞지 않고 앞 70%, 뒤 30%로 분리
split_index = int(len(df) * 0.7)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_test = y.iloc[split_index:].copy()


print("전체 특징 개수:")
print(len(feature_cols))

print("\n학습 데이터 크기:")
print(X_train.shape)

print("\n검증 데이터 크기:")
print(X_test.shape)

print("\n학습 기간:")
print(df["timestamp"].iloc[0])
print("~")
print(df["timestamp"].iloc[split_index - 1])

print("\n검증 기간:")
print(df["timestamp"].iloc[split_index])
print("~")
print(df["timestamp"].iloc[-1])

# ============================================================
# 9. Z-score 기준 모델
# ============================================================

from sklearn.preprocessing import StandardScaler

# 기준 모델에서는 원본 센서값만 사용
# rolling, diff, lag 특징은 사용하지 않음
z_scaler = StandardScaler()


# 학습 데이터의 센서값으로 평균과 표준편차 계산
z_train = z_scaler.fit_transform(X_train[sensor_cols])


# 검증 데이터에 학습 데이터 기준 적용
z_test = z_scaler.transform(X_test[sensor_cols])


# 각 시점에서 가장 큰 절댓값 Z-score를 이상 점수로 사용
z_train_score = np.max(np.abs(z_train), axis=1)

z_test_score = np.max(np.abs(z_test), axis=1)


# 학습 데이터 상위 1%를 이상 기준으로 설정
z_threshold = np.quantile(z_train_score, 0.99)


# 검증 데이터 이상 여부
z_test_anomaly = (z_test_score > z_threshold).astype(int)


print("Z-score 기준값:")
print(z_threshold)

print("\nZ-score 이상으로 판단된 검증 데이터 개수:")
print(z_test_anomaly.sum())

print("\n검증 데이터 전체 개수:")
print(len(z_test_anomaly))

# ============================================================
# 10. Z-score 결과 평가
# ============================================================

from sklearn.metrics import confusion_matrix, classification_report

# NORMAL은 정상(0)
# RECOVERING과 BROKEN은 비정상(1)으로 변환
y_test_binary = (y_test != "NORMAL").astype(int)


print("Z-score 혼동행렬:")
print(confusion_matrix(y_test_binary, z_test_anomaly))


print("\nZ-score Precision / Recall / F1:")
print(
    classification_report(
        y_test_binary,
        z_test_anomaly,
        target_names=["NORMAL", "ABNORMAL"],
        zero_division=0,
    )
)

# ============================================================
# 11. Isolation Forest 모델
# ============================================================

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

# 센서값, missing 정보, rolling, diff, lag 특징을 함께 사용
# 센서별 값의 범위가 다르므로 스케일 조정
robust_scaler = RobustScaler()

X_train_scaled = robust_scaler.fit_transform(X_train)
X_test_scaled = robust_scaler.transform(X_test)


# 라벨 없이 이상 패턴 학습
isolation_model = IsolationForest(
    n_estimators=200, contamination="auto", random_state=42, n_jobs=-1
)

isolation_model.fit(X_train_scaled)


# 이상 점수 계산
# decision_function 값이 작을수록 이상하므로 음수 부호를 붙임
if_train_score = -isolation_model.decision_function(X_train_scaled)

if_test_score = -isolation_model.decision_function(X_test_scaled)


# 학습 데이터 상위 1%를 이상 기준으로 설정
if_threshold = np.quantile(if_train_score, 0.99)


# 검증 데이터 이상 여부
if_test_anomaly = (if_test_score > if_threshold).astype(int)


print("Isolation Forest 학습 완료")

print("\nIsolation Forest 기준값:")
print(if_threshold)

print("\nIsolation Forest 이상 개수:")
print(if_test_anomaly.sum())

print("\n검증 데이터 전체 개수:")
print(len(if_test_anomaly))

# ============================================================
# 12. Isolation Forest 결과 평가
# ============================================================

print("Isolation Forest 혼동행렬:")
print(confusion_matrix(y_test_binary, if_test_anomaly))


print("\nIsolation Forest Precision / Recall / F1:")
print(
    classification_report(
        y_test_binary,
        if_test_anomaly,
        target_names=["NORMAL", "ABNORMAL"],
        zero_division=0,
    )
)


# ============================================================
# 13. 이상 후보 사건 묶기
# ============================================================

# 검증 구간의 원본 정보와 모델 결과를 하나로 구성
result = df.iloc[split_index:].copy()

result["if_score"] = if_test_score
result["if_anomaly"] = if_test_anomaly


# 이상으로 판단된 시점만 선택
anomaly_points = result[result["if_anomaly"] == 1].copy()


if len(anomaly_points) > 0:

    # 이전 이상 시점과의 시간 차이 계산
    time_gap = anomaly_points["timestamp"].diff()

    # 이전 이상 시점과 30분보다 차이가 크면
    # 새로운 사건으로 판단
    new_event = time_gap.isna() | (time_gap > pd.Timedelta(minutes=30))

    # 사건 번호 생성
    anomaly_points["event_id"] = new_event.cumsum()

    # 사건별 요약
    event_summary = (
        anomaly_points.groupby("event_id")
        .agg(
            event_start=("timestamp", "min"),
            event_end=("timestamp", "max"),
            anomaly_count=("timestamp", "count"),
            max_score=("if_score", "max"),
        )
        .reset_index()
    )

    # 사건 지속 시간 계산
    event_summary["duration_minutes"] = (
        event_summary["event_end"] - event_summary["event_start"]
    ).dt.total_seconds() / 60

    print("이상 후보 사건 요약:")
    print(event_summary)

else:
    event_summary = pd.DataFrame()

    print("탐지된 이상 후보 사건이 없습니다.")

# ============================================================
# 14. 오탐·미탐 분석
# ============================================================

# 정상은 0, RECOVERING/BROKEN은 1
result["actual_abnormal"] = (result["machine_status"] != "NORMAL").astype(int)


# 오탐:
# 모델은 이상이라고 했지만 실제는 정상
false_positive = result[(result["if_anomaly"] == 1) & (result["actual_abnormal"] == 0)]


# 미탐:
# 실제는 비정상인데 모델이 정상이라고 판단
false_negative = result[(result["if_anomaly"] == 0) & (result["actual_abnormal"] == 1)]


print("오탐 개수:")
print(len(false_positive))

print("\n미탐 개수:")
print(len(false_negative))

print("\n오탐 상태 분포:")
print(false_positive["machine_status"].value_counts())

print("\n미탐 상태 분포:")
print(false_negative["machine_status"].value_counts())

print("\n오탐 사례:")
print(false_positive[["timestamp", "if_score", "machine_status"]].head(10))

print("\n미탐 사례:")
print(false_negative[["timestamp", "if_score", "machine_status"]].head(10))

# ============================================================
# 15. 시간 흐름에 따른 이상 점수 그래프
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))


# Isolation Forest 이상 점수
plt.plot(
    result["timestamp"],
    result["if_score"],
    color="steelblue",
    linewidth=0.7,
    label="Isolation Forest score",
)


# 이상 판정 기준선
plt.axhline(
    y=if_threshold,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Anomaly threshold",
)


# 실제 비정상 시점
actual_abnormal_points = result[result["actual_abnormal"] == 1]

plt.scatter(
    actual_abnormal_points["timestamp"],
    actual_abnormal_points["if_score"],
    color="red",
    s=18,
    label="Actual abnormal",
)


# 모델이 이상으로 판단한 시점
predicted_anomaly_points = result[result["if_anomaly"] == 1]

plt.scatter(
    predicted_anomaly_points["timestamp"],
    predicted_anomaly_points["if_score"],
    color="orange",
    s=12,
    label="Detected anomaly",
)


plt.title("Isolation Forest Anomaly Score Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 16. 이상에 영향을 준 센서 확인
# ============================================================

# 검증 구간의 센서별 절댓값 Z-score
abs_z_test = pd.DataFrame(abs(z_test), columns=sensor_cols, index=result.index)


# 모델이 이상으로 판단한 시점
detected_mask = result["if_anomaly"] == 1

# 모델이 정상으로 판단한 시점
normal_mask = result["if_anomaly"] == 0


# 이상 시점과 정상 시점의 센서별 평균 비교
detected_mean = abs_z_test.loc[detected_mask].mean()

normal_mean = abs_z_test.loc[normal_mask].mean()


# 이상 시점에서 더 크게 변한 센서 순서
sensor_effect = (detected_mean - normal_mean).sort_values(ascending=False)


print("이상 탐지에 크게 영향을 준 센서:")
print(sensor_effect.head(10))

# ============================================================
# 17. 모델 성능 비교
# ============================================================

from sklearn.metrics import precision_recall_fscore_support

# Z-score 성능
z_precision, z_recall, z_f1, _ = precision_recall_fscore_support(
    y_test_binary, z_test_anomaly, average="binary", zero_division=0
)


# Isolation Forest 성능
if_precision, if_recall, if_f1, _ = precision_recall_fscore_support(
    y_test_binary, if_test_anomaly, average="binary", zero_division=0
)


comparison = pd.DataFrame(
    {
        "model": ["Z-score", "Isolation Forest"],
        "precision": [z_precision, if_precision],
        "recall": [z_recall, if_recall],
        "f1": [z_f1, if_f1],
    }
)


print("모델 성능 비교:")
print(comparison.set_index("model").round(3))

# ============================================================
# 18. BROKEN 상태 별도 평가
# ============================================================

# BROKEN만 실제 고장으로 정의
y_test_broken = (y_test == "BROKEN").astype(int)


print("Z-score의 BROKEN 탐지 결과:")
print(confusion_matrix(y_test_broken, z_test_anomaly))

print(
    classification_report(
        y_test_broken,
        z_test_anomaly,
        target_names=["NOT_BROKEN", "BROKEN"],
        zero_division=0,
    )
)


print("\nIsolation Forest의 BROKEN 탐지 결과:")
print(confusion_matrix(y_test_broken, if_test_anomaly))

print(
    classification_report(
        y_test_broken,
        if_test_anomaly,
        target_names=["NOT_BROKEN", "BROKEN"],
        zero_division=0,
    )
)


# ============================================================
# 19. 이상 후보 사건과 실제 BROKEN 시점 비교
# ============================================================

if len(event_summary) > 0:

    event_check = event_summary.copy()

    broken_times = (
        result.loc[result["machine_status"] == "BROKEN", "timestamp"]
        .sort_values()
        .tolist()
    )

    recovering_times = (
        result.loc[result["machine_status"] == "RECOVERING", "timestamp"]
        .sort_values()
        .tolist()
    )

    broken_overlap = []
    recovering_overlap = []

    for _, event in event_check.iterrows():

        # 사건 전후 30분까지 확인
        check_start = event["event_start"] - pd.Timedelta(minutes=30)

        check_end = event["event_end"] + pd.Timedelta(minutes=30)

        has_broken = any(check_start <= t <= check_end for t in broken_times)

        has_recovering = any(check_start <= t <= check_end for t in recovering_times)

        broken_overlap.append(has_broken)
        recovering_overlap.append(has_recovering)

    event_check["near_broken"] = broken_overlap
    event_check["near_recovering"] = recovering_overlap

    print("이상 후보 사건과 실제 상태 비교:")
    print(event_check)

else:
    print("비교할 이상 후보 사건이 없습니다.")


# ============================================================
# 20. 고장 전조 탐지 시간 확인
# ============================================================

broken_times = (
    result.loc[result["machine_status"] == "BROKEN", "timestamp"].sort_values().tolist()
)

if len(event_summary) > 0 and len(broken_times) > 0:

    for broken_time in broken_times:

        # BROKEN 이전에 끝난 이상 후보 사건만 선택
        previous_events = event_summary[event_summary["event_end"] <= broken_time]

        if len(previous_events) > 0:

            # 고장 직전에 발생한 후보 사건 선택
            last_event = previous_events.iloc[-1]

            lead_time = (broken_time - last_event["event_end"]).total_seconds() / 60

            print("\nBROKEN 발생 시각:")
            print(broken_time)

            print("직전 이상 후보 종료 시각:")
            print(last_event["event_end"])

            print("고장 전 탐지 간격(분):")
            print(lead_time)

        else:
            print(f"\n{broken_time} 이전에 탐지된 이상 후보가 없습니다.")

else:
    print("고장 전조 시간을 계산할 조건이 부족합니다.")


# 확인해야할 결과
# 1. Z-score와 Isolation Forest 중 F1이 높은 모델
# 2. 실제 BROKEN을 탐지했는지
# 3. BROKEN보다 먼저 이상 후보가 나왔는지
# 4. 오탐·미탐 개수
# 5. 이상에 크게 영향을 준 센서
# 6. 정비·비가동으로 보이는 0값 구간이 있었는지

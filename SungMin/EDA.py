import pandas as pd
import numpy as np

# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv("data/sensor.csv")

print("원본 데이터 크기:", df.shape)


# ============================================================
# 2. timestamp 변환 + 시간순 정렬
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("\n시간 범위")
print(df["timestamp"].min(), "~", df["timestamp"].max())


# ============================================================
# 3. 센서 컬럼 추출
# ============================================================

sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

print("\n센서 개수:", len(sensor_cols))


# ============================================================
# 4. 전체가 결측치인 센서 확인 및 제거
#    ex) sensor_15
# ============================================================

all_missing_cols = [col for col in sensor_cols if df[col].isna().all()]

print("\n전체 결측 센서")
print(all_missing_cols)

df = df.drop(columns=all_missing_cols)

# 센서 목록 다시 갱신
sensor_cols = [col for col in sensor_cols if col not in all_missing_cols]

print("제거 후 센서 개수:", len(sensor_cols))


# ============================================================
# 5. 0값 확인
#    ★ 0은 결측치로 처리하지 않음
# ============================================================

zero_ratio = (df[sensor_cols] == 0).mean().mul(100).sort_values(ascending=False)

print("\n센서별 0 비율 TOP 10")
print(zero_ratio.head(10))

# 0은 설비 정지 / 운전상태 / 실제 측정값일 가능성이 있으므로
# 그대로 유지한다.


# ============================================================
# 6. 결측률 확인
# ============================================================

missing_ratio_before = (
    df[sensor_cols].isna().mean().mul(100).sort_values(ascending=False)
)

print("\n전처리 전 결측률")
print(missing_ratio_before[missing_ratio_before > 0])


# ============================================================
# 7. 결측 발생 여부를 별도 변수로 저장
#
# 나중에 "결측이 발생한 것 자체"가
# 특정 설비 상태와 관련 있는지 확인할 수 있음
# ============================================================

missing_flag_cols = []

for col in sensor_cols:

    if df[col].isna().any():

        flag_col = f"{col}_missing"

        df[flag_col] = df[col].isna().astype(int)

        missing_flag_cols.append(flag_col)

print("\n결측 Flag 생성")
print(missing_flag_cols)


# ============================================================
# 8. 유일값 개수 확인
#    값 종류가 매우 적은 센서는
#    상태형 / 이산형 변수 후보
# ============================================================

nunique = df[sensor_cols].nunique().sort_values()

print("\n유일값 개수가 적은 센서")
print(nunique.head(15))


# 우선 unique 값이 20개 이하이면
# 이산형 센서 후보로 분류
#
# ★ 20은 절대적인 기준이 아님
# EDA를 위한 임시 기준
discrete_cols = nunique[nunique <= 20].index.tolist()

continuous_cols = [col for col in sensor_cols if col not in discrete_cols]

print("\n이산형 후보")
print(discrete_cols)

print("\n연속형 센서 개수")
print(len(continuous_cols))


# ============================================================
# 9. timestamp를 index로 설정
#
# time interpolation을 사용하기 위해 필요
# ============================================================

df = df.set_index("timestamp")


# ============================================================
# 10. 연속형 센서 결측 처리
#
# 5분 이하의 짧은 결측구간만 시간 기반 보간
#
# 긴 결측구간은 억지로 채우지 않음
# ============================================================

if continuous_cols:

    df[continuous_cols] = df[continuous_cols].interpolate(
        method="time",
        limit=5,
        limit_direction="both",  # 양방향 최대 5개 연속된 애까지 채움, 시간을 기준으로
    )


# ============================================================
# 11. 이산형 / 상태형 후보
#
# 선형보간을 하면
# 0 → 1 사이에 0.5 같은 이상한 값이 생길 수 있으므로
# 최대 5분까지만 이전 값을 사용
# ============================================================

if discrete_cols:

    df[discrete_cols] = df[discrete_cols].ffill(
        limit=5
    )  # 앞에있는걸로 채워라 5개연속까지는~


# ============================================================
# 12. timestamp 다시 컬럼으로 복구
# ============================================================

df = df.reset_index()


# ============================================================
# 13. 전처리 후 결측치 확인
# ============================================================

missing_ratio_after = (
    df[sensor_cols].isna().mean().mul(100).sort_values(ascending=False)
)

print("\n전처리 후 남은 결측률")

print(missing_ratio_after[missing_ratio_after > 0])


# ============================================================
# 14. 결측률이 너무 높은 센서는
#     첫 번째 모델에서 제외
#
# 여기서는 임시로 10% 기준
# ============================================================

MISSING_THRESHOLD = 10

model_sensor_cols = missing_ratio_after[
    missing_ratio_after < MISSING_THRESHOLD
].index.tolist()

print("\n모델에 사용할 센서 개수")
print(len(model_sensor_cols))

print("\n모델 제외 센서")
print([col for col in sensor_cols if col not in model_sensor_cols])


# ============================================================
# 15. 모델용 DataFrame 생성
#
# 원본 df는 유지하고
# 모델용 데이터만 따로 만든다.
# ============================================================

model_df = df[["timestamp"] + model_sensor_cols + missing_flag_cols].copy()


# ============================================================
# 16. 모델용 데이터에서 아직 NaN이 남은 행만 제거
#
# ★ 원본 df의 행을 삭제하는 것이 아님
# ★ Isolation Forest 입력용 데이터에서만 제거
# ============================================================

before_rows = len(model_df)

model_df = model_df.dropna(subset=model_sensor_cols).reset_index(drop=True)

after_rows = len(model_df)

print("\n모델 데이터 행 수")
print("처리 전 :", before_rows)
print("처리 후 :", after_rows)
print("제거 :", before_rows - after_rows)


# ============================================================
# 17. 최종 X 생성
#
# timestamp는 모델 입력에서 제외
# ============================================================

feature_cols = model_sensor_cols + missing_flag_cols

X = model_df[feature_cols].copy()


print("\n최종 X shape")
print(X.shape)

print("\nNaN 개수")
print(X.isna().sum().sum())


# ============================================================
# 18. 최종 확인
# ============================================================

print("\n최종 데이터 정보")
print(model_df.info())

print("\n최종 사용 변수")
print(feature_cols)


print(df[feature_cols].isna().sum())


# 여기서부터 아이솔레이트트리!
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================================
# 19. 스케일링
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 20. Isolation Forest 모델 생성
# ============================================================

iso_model = IsolationForest(
    n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1
)


# ============================================================
# 21. 모델 학습 + 이상치 예측
# ============================================================

model_df["anomaly"] = iso_model.fit_predict(X_scaled)


# ============================================================
# 22. 이상치 점수 저장
# ============================================================

model_df["anomaly_score"] = iso_model.decision_function(X_scaled)


# ============================================================
# 23. 결과 확인
# ============================================================

print("\nIsolation Forest 결과")
print(model_df["anomaly"].value_counts())

print("\n이상 비율")
print(model_df["anomaly"].value_counts(normalize=True) * 100)


# ============================================================
# 24. 탐지된 이상 데이터 확인
# ============================================================

anomaly_df = model_df[model_df["anomaly"] == -1]

print("\n탐지된 이상 데이터 개수")
print(len(anomaly_df))

print("\n이상 데이터 예시")
print(anomaly_df[["timestamp", "anomaly", "anomaly_score"]].head(20))

import pandas as pd

# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv("data/sensor.csv")


# ============================================================
# 2. 불필요한 컬럼 제거
# ============================================================

# CSV 저장 과정에서 생성된 인덱스 컬럼 제거
df = df.drop(columns=["Unnamed: 0"], errors="ignore")


# ============================================================
# 3. timestamp 변환 및 시간순 정렬
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 4. 센서 컬럼 설정
# ============================================================

sensor_cols = [col for col in df.columns if col.startswith("sensor_")]


# ============================================================
# 5. 센서별 결측률 확인
# ============================================================

missing_rate = df[sensor_cols].isna().mean() * 100

print("=== 센서별 결측률 TOP 10 ===")
print(missing_rate.sort_values(ascending=False).head(10))


# ============================================================
# 6. 결측률 30% 이상 센서 제거
# ============================================================

high_missing = missing_rate[missing_rate >= 30].index.tolist()

print("\n=== 결측률 30% 이상 제거 센서 ===")
print(high_missing)

df = df.drop(columns=high_missing)


# ============================================================
# 7. 센서 컬럼 다시 설정
# ============================================================

sensor_cols = [col for col in df.columns if col.startswith("sensor_")]


# ============================================================
# 8. 값이 변하지 않는 센서 확인 및 제거
# ============================================================

unique_count = df[sensor_cols].nunique()

constant_sensors = unique_count[unique_count <= 1].index.tolist()

print("\n=== 값이 변하지 않는 센서 ===")
print(constant_sensors)

df = df.drop(columns=constant_sensors)


# ============================================================
# 9. 센서 컬럼 다시 설정
# ============================================================

sensor_cols = [col for col in df.columns if col.startswith("sensor_")]


# ============================================================
# 10. 최대 연속 결측 길이 확인
# ============================================================


def max_consecutive_nan(series):
    is_nan = series.isna()

    groups = (is_nan != is_nan.shift()).cumsum()

    consecutive = is_nan.groupby(groups).sum()

    return consecutive.max()


max_nan_length = df[sensor_cols].apply(max_consecutive_nan)

print("\n=== 최대 연속 결측 길이 TOP 10 ===")
print(max_nan_length.sort_values(ascending=False).head(10))


# ============================================================
# 11. 짧은 결측 구간 보간
# ============================================================

# 데이터가 1분 간격이므로
# 최대 60개의 연속 결측값까지만 선형 보간
df[sensor_cols] = df[sensor_cols].interpolate(
    method="linear", limit=60, limit_area="inside"
)


# ============================================================
# 12. 보간 후 남은 결측치 확인
# ============================================================

print("\n=== 보간 후 남은 결측치 TOP 10 ===")
print(df[sensor_cols].isna().sum().sort_values(ascending=False).head(10))


# ============================================================
# 13. 장기 결측이 남아있는 행 제외
# ============================================================

missing_rows = df[sensor_cols].isna().any(axis=1)

original_rows = len(df)
removed_rows = missing_rows.sum()

clean_df = df.loc[~missing_rows].copy()

remaining_rows = len(clean_df)
retention_rate = remaining_rows / original_rows * 100


# ============================================================
# 14. 최종 데이터 품질 확인
# ============================================================

print("\n========== 최종 전처리 결과 ==========")

print("\n[행 처리 결과]")
print("처리 전:", original_rows)
print("제외된 행:", removed_rows)
print("처리 후:", remaining_rows)
print(f"데이터 유지율: {retention_rate:.2f}%")

print("\n[데이터 크기]")
print(clean_df.shape)

print("\n[전체 결측치]")
print(clean_df.isna().sum().sum())

print("\n[중복 행]")
print(clean_df.duplicated().sum())

print("\n[timestamp 중복]")
print(clean_df["timestamp"].duplicated().sum())

print("\n[데이터 기간]")
print(clean_df["timestamp"].min(), "~", clean_df["timestamp"].max())

print("\n[최종 컬럼]")
print(clean_df.columns)


# ============================================================
# 15. 전처리 완료 데이터 저장
# ============================================================

clean_df.to_csv("data/sensor_preprocessed.csv", index=False)

print("\n전처리 데이터 저장 완료")
print("data/sensor_preprocessed.csv")


# 데이터 전처리 브리핑

# 원본 데이터: 220,320행, 55개 컬럼으로 구성
# 불필요한 Unnamed: 0 컬럼 제거
# timestamp 날짜·시간 형식 변환 및 시간순 정렬
# 센서별 결측률 확인 후 sensor_15 100%, sensor_50 약 35% 결측으로 제거
# 일부 센서(sensor_00, sensor_06, sensor_07, sensor_08, sensor_09, sensor_51)의 연속 결측 시간 추가 확인
# 해당 센서에서 약 3~10일 정도의 장기 연속 결측 확인
# 특히 sensor_00, sensor_06, sensor_07, sensor_08, sensor_09에서 6월 말~7월 초 결측 구간이 서로 겹치는 패턴 확인
# 장기 결측까지 보간할 경우 실제 측정되지 않은 값이 과도하게 생성될 가능성이 있어 짧은 결측만 제한적으로 선형 보간
# 보간 후에도 장기 결측이 남아 있는 행 제외
# 최종 데이터: 196,507행 × 52열
# 전체 데이터의 89.19% 유지
# 최종 결측치 0개, 중복 행 0개, 중복 timestamp 0개
# 전처리 완료 데이터 sensor_preprocessed.csv로 별도 저장
# machine_status는 비지도 이상 탐지에는 사용하지 않고 최종 결과 검증용으로 유지


# 참고해야할 내용(피티 피셜)

# 왜 결측률 30%를 기준으로 했는지: 30%가 정해진 공식 기준은 아니고,
# 이번 짧은 탐색 프로젝트에서 결측이 너무 많은 센서를 과도하게 보간하지 않기 위해 설정한 기준.
# 이에 따라 sensor_15(100%), sensor_50(약 35%) 제거.

# 왜 sensor_51은 결측률이 약 7%인데 제거하지 않았는지: 결측률만으로 센서를 삭제하지 않고 연속 결측 패턴까지 확인.
# sensor_51은 약 10.7일의 긴 결측이 있었지만 나머지 기간의 실제 측정값까지 버리지 않기 위해 센서 자체는 유지하고 결측이 남은 행을 제외.

# 왜 sensor_00, 06~09를 제거하지 않았는지: 이 센서들은 약 3~4일의 긴 결측이 있었지만,
# 특히 6월 말~7월 초에 결측 시점이 서로 겹치는 패턴 확인. 개별 센서 자체의 문제라고 단정하기 어려워 센서 전체를 삭제하지 않고 유지.

# 왜 모든 결측치를 보간하지 않았는지: 며칠 동안 측정되지 않은 수천 개의 값을 보간하면 실제로 존재하지 않는 센서값을 대량으로 만들어낼 수 있음.
# 따라서 짧은 결측만 제한적으로 선형 보간하고 장기 결측은 보간하지 않음.

# 60분 보간 기준도 절대적인 기준이 아님: 데이터가 1분 간격이기 때문에 이번 탐색에서 짧은 공백을 제한적으로 보완하기 위한 기준으로 60개를 사용.
# 특히 limit=60은 긴 결측 구간 전체를 완전히 건드리지 않는 옵션은 아니라는 점도 알고 있어야 함.

# 23,813행을 왜 제외했는지: 짧은 결측 보간 후에도 하나 이상의 센서에 NaN이 남아 있는 행을 제외.
# 전체 220,320행 중 196,507행, 89.19% 유지. 즉 약 10.81%를 제외.

# 시간 데이터가 완전히 연속적이지 않게 됐다는 점은 꼭 시계열 담당자에게 전달: 원래는 1분 간격 데이터지만
# 23,813행을 제외했기 때문에 sensor_preprocessed.csv에서는 일부 timestamp 사이에 시간 공백이 생김.
# 따라서 시계열 담당자가 rolling, lag, 차분 등을 만들 때 모든 인접 행을 무조건 1분 차이라고 생각하면 안 됨.

# machine_status를 왜 남겼는지: NORMAL / RECOVERING / BROKEN 라벨은 삭제하지 않고 최종 검증용으로 보존.
# 단, 센서 선택·전처리·Isolation Forest 학습에서는 사용하지 않고 마지막에 탐지 결과와 실제 상태를 비교할 때만 사용.

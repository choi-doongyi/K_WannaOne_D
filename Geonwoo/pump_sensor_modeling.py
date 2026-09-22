import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
from sklearn.dummy import DummyClassifier


def set_korean_font():
    font_candidates = [
        "Malgun Gothic",  # Windows
        "AppleGothic",  # macOS
        "Apple SD Gothic Neo",  # macOS
        "NanumGothic",  # Linux / 설치된 경우
        "Noto Sans CJK KR",  # Linux / 설치된 경우
        "Noto Sans KR",  # 설치된 경우
    ]

    available_fonts = {font.name for font in fm.fontManager.ttflist}

    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return font

    plt.rcParams["axes.unicode_minus"] = False
    print("사용 가능한 한글 폰트를 찾지 못했습니다.")
    return None


# sns.set_theme(style="whitegrid") # 필요하면 주석 해제
selected_font = set_korean_font()
print("사용 폰트:", selected_font)

df = pd.read_csv("data/sensor.csv")
print(df.shape)
print(df.isna().sum().value_counts())

# 불필요한 인덱스 컬럼 제거
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# 시간 컬럼 변환 및 정렬
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 결측률 확인
missing_rate = df.isna().mean().sort_values(ascending=False)
print(missing_rate)

# 전체가 결측인 센서 제거
all_missing = [
    col for col in df.columns
    if col.startswith("sensor_") and df[col].isna().all()
]

df = df.drop(columns=all_missing)

# 센서 컬럼 목록
sensor_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

# 시간 순서 기반 결측 보간
df[sensor_columns] = (
    df[sensor_columns]
    .interpolate(method="linear", limit_direction="both")
)

print(df.info())

sensor_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

missing_by_sensor = (
    df[sensor_columns]
      .isna()
      .sum()
      .sort_values(ascending=False)
)

print(missing_by_sensor)

zero_count_by_column = (
    df.eq(0)
      .sum()
      .sort_values(ascending=False)
)

print(zero_count_by_column)

import pandas as pd

df["timestamp"] = pd.to_datetime(df["timestamp"])

sensor_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

df["hour"] = df["timestamp"].dt.hour

# 시간대별 센서 0 비율(%)
zero_ratio_by_hour = (
    df.groupby("hour")[sensor_columns]
      .apply(lambda x: x.eq(0).mean() * 100)
)

print(zero_ratio_by_hour.round(2))

summary = pd.DataFrame({
    "peak_hour": zero_ratio_by_hour.idxmax(),
    "peak_zero_ratio(%)": zero_ratio_by_hour.max(),
    "overall_zero_ratio(%)": (
        df[sensor_columns].eq(0).mean() * 100
    )
})

summary = summary.sort_values(
    "peak_zero_ratio(%)",
    ascending=False
)

print(summary.round(2))

zero_sensors = [
    "sensor_19", "sensor_13", "sensor_18", "sensor_17",
    "sensor_22", "sensor_37", "sensor_25", "sensor_12",
    "sensor_24", "sensor_05", "sensor_11", "sensor_16",
    "sensor_27", "sensor_23", "sensor_35", "sensor_10",
    "sensor_20", "sensor_00", "sensor_07", "sensor_01",
    "sensor_30", "sensor_09",
]

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

zero_ratio_by_hour = (
    df.groupby("hour")[zero_sensors]
      .apply(lambda x: x.eq(0).mean() * 100)
)

print(zero_ratio_by_hour.round(2))


zero_ratio_by_status = (
    df.groupby("machine_status")[zero_sensors]
      .apply(lambda x: x.eq(0).mean() * 100)
)

print("\nmachine_status별 0 발생 비율(%):")
print(zero_ratio_by_status.round(2))

run_results = []

for sensor in zero_sensors:
    is_zero = df[sensor].eq(0)

    # 0/비0 상태가 바뀔 때마다 그룹 생성
    group_id = (
        is_zero
        .ne(is_zero.shift(fill_value=False))
        .cumsum()
    )

    # 연속된 0 구간 길이 계산
    zero_runs = (
        is_zero[is_zero]
        .groupby(group_id)
        .sum()
        .astype(int)
    )

    # 같은 길이의 구간이 몇 번 발생했는지 계산
    run_frequency = zero_runs.value_counts()

    for run_length, count in run_frequency.items():
        run_results.append({
            "sensor": sensor,
            "zero_run_length": run_length,
            "occurrence_count": count
        })

long_zero_runs = pd.DataFrame(run_results)

# 긴 구간부터 출력
long_zero_runs = long_zero_runs.sort_values(
    ["zero_run_length", "occurrence_count"],
    ascending=False
)

print("\n연속된 0 구간:")
print(long_zero_runs.head(50))
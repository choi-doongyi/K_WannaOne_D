import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  # 그래프
import platform
import seaborn as sns  # 씨본 그래프
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # 정확도
from sklearn.dummy import DummyClassifier  # 베이스라인 dummy
from sklearn.metrics import confusion_matrix  # 혼동행렬
from sklearn.metrics import ConfusionMatrixDisplay  # 혼동행렬 heatmap
from sklearn.metrics import precision_score  # 정밀도
from sklearn.metrics import recall_score  # 재현율
from sklearn.metrics import f1_score, precision_score, recall_score  # 조화평균
from sklearn.metrics import classification_report  # 한눈에
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df1 = pd.read_csv("data/sensor.csv")
df2 = pd.read_csv("data/sensor.csv")

# 원본 데이터 확인
df = df1.copy()

df.info()
print(df.shape)
print(df.describe())
print("센서별 결측 개수:", df.isna().sum())

print("=====================")

# timestamp index로 정렬
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

# 데이터 전처리

# 연속된 결측치 개수
print("연속된 결측치 개수")
print("=====================")
result = {}

for col in df.columns:
    is_na = df[col].isna()

    # 결측/비결측 상태가 바뀔 때마다 그룹 번호 생성
    group = is_na.ne(is_na.shift()).cumsum()

    # 결측치 구간만 골라서 각 구간의 길이 계산
    na_run_lengths = is_na.groupby(group).sum()

    # 0이 아닌 값만 남기기
    na_run_lengths = na_run_lengths[na_run_lengths > 0]

    result[col] = list(na_run_lengths)

print(result)

print("=====================")
# 전부 결측이거나 결측치 비율이 높은 컬럼 제거 15, 00, 50, 51

# 제거할 센서 컬럼
drop_cols = ["sensor_15", "sensor_00", "sensor_50", "sensor_51"]

# 실제 존재하는 컬럼만 제거
df = df.drop(columns=drop_cols, errors="ignore")

# 남은 센서 컬럼 확인
sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

print("남은 센서 개수:", len(sensor_cols))
print(sensor_cols)
print(df.isna().mean().sort_values(ascending=False).head(10))
print("=====================")

## 다운샘플링으로 추세를 파악해보려함
df = df[2:-1]  # time stamp와 machine_status 열을 제외한 센서컬럼들
df_5min_m = df.resample("5min").mean(numeric_only=True)  # 5분 단위

df_10min_m = df.resample("10min").mean(numeric_only=True)  # 10분

df_h_m = df.resample("h").mean(numeric_only=True)  # 1시간

print(df_5min_m)


## 이동평균 · 이동표준편차 · 변화율 · 시차 변수 중 3종 이상 생성  (시계열 담당 과제 의무)

rol_mean = df_5min_m.rolling(window=5).mean()  # 다운샘플링한 데이터의 이동평균
rol_std = df_5min_m.rolling(window=5).std()  # 이동표준편차
pct = df_5min_m.pct_change()  # 변화율
print("이동평균:", rol_mean)
print("이동표준편차:", rol_std)
print("변화율:", pct)

####
# 각 컬럼별 리샘플링 후 시각화로 추세 파악

# for c in df.columns[2:-1]:
#    plt.figure(figsize=(12, 5))
#    plt.plot(df_5min_m.index, df_5min_m[c], label=c, alpha=0.4)
#    plt.plot(rol_mean.index, rol_mean[c], label=c, linewidth=2)
#    plt.xlabel("Timestamp")
#    plt.title("5min Rolling Mean")
#    plt.legend()
#    plt.grid(True)
#    plt.show()
print("=====================")

sensor_cols = df.columns[1:-1]  # 기존에 사용하던 컬럼 범위

group_size = 6

for i in range(0, len(sensor_cols), group_size):
    cols = sensor_cols[i : i + group_size]

    fig, axes = plt.subplots(
        nrows=len(cols), ncols=1, figsize=(14, 2.5 * len(cols)), sharex=True
    )

    # cols가 1개만 남는 경우 axes가 리스트가 아닐 수 있어서 처리
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.plot(df_h_m.index, df_h_m[c], label=c, alpha=0.7)
        ax.set_title(c)
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Timestamp")

    fig.suptitle(f"5min Rolling Mean Sensors {i + 1} ~ {i + len(cols)}", fontsize=16)

    plt.tight_layout()
    plt.show()

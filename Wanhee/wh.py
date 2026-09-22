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
print(df.isna().sum())

print("=====================")

# timestamp index로 정렬
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

# 데이터 전처리


## 규모(3~4달)에 비해 측정단위(1분)이 너무 작기 때문에 다운샘플링으로 추세를 파악해보려함
df = df[2:-1]  # machine_status 열을 제외함
df_h_m = df.resample("h").mean(numeric_only=True)  # 시간 단위로 리샘플링 후 평균
# 시간대 구분 — 24시간 안의 운영 패턴
df_D_m = df.resample("D").mean(numeric_only=True)  # 일 단위
# 흐름 부각 — 일별 평균이 추세를 또렷하게
df_W_m = df.resample("W").mean(numeric_only=True)  # 주 단위
# 큰 그림 — 시간, 일, 주 단위 평균으로 장기 변화 관찰
print(df_h_m)

## 이동평균 · 이동표준편차 · 변화율 · 시차 변수 중 3종 이상 생성  (시계열 담당 과제 의무)

rol_mean = df_D_m.rolling(window=5).mean()  # 다운샘플링한 데이터의 이동평균
rol_std = df_D_m.rolling(window=5).std()  # 이동표준편차
pct = df_D_m.pct_change()  # 변화율
print(rol_mean)


####

# df["failure_soon"] = ###선별한 이상후보구간에 맞는경우

feature_cols = df.columns[2:-1]
X = df[feature_cols]
y = df["failure_soon"]


for c in df.columns[2:-1]:
    plt.figure(figsize=(12, 5))
    plt.plot(df_D_m.index, df_D_m[c], label=c, alpha=0.4)
    plt.plot(rol_mean.index, rol_mean[c], label=c, linewidth=2)

    plt.xlabel("Timestamp")
    plt.title("Day Mean and Rolling Mean")
    plt.legend()
    plt.grid(True)
    plt.show()

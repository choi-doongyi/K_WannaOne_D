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


df = df1.copy()

df.info()
print(df.shape)
print(df.describe())
print(df.isna().sum())

## 난점 대응 계획과 작업 백로그 작성
# 컬럼의 수가 많고 이름으로 기능을 알 수 없다
# 무엇이 중요한 정보인지 한 눈에 알기 어렵고 데이터 전처리 과정에서 중요한 정보를 놓치거나
# 중요하지 않은 정보에 집중하면 다음 과정이 무의미해진다
# wh의견 corr을 이용하여 각 컬럼별 상관을 파악하고 상관이 높은 센서끼리의
# 상관이 깨지는 순간을 이상신호를 보는데 우선시 하는건 어떨까 (다 같은펌프설비 센서라서 상관계수가
# 너무 높거나 낮게 나옴 + 컬럼의 수가 너무 많아 눈에 들어오지 않음)


# 데이터의 수집 기간(월 단위)에 비해 측정간격(1분 단위)가 작아 데이터의 흐름이 한눈에 들어오지 않는다
# wh의견 resaple을 통해 일(D)이나 주(W)단위로 묶어서 추세파악을 한다

#### corr = df[df.columns[2:-1]].corr().unstack()  # 컬럼들의 상관행렬을 일렬로 만듬
#### high_corr = corr[corr < 1].abs().sort_values(ascending=False) # 자기자신(1)을 제외한
# 상관계수 높은순으로 정렬
#### print(high_corr.head())  ####  상관계수가 높은경우가 너무 많음  보류
print("=====================")

#### 시계열 분석용도
# timestamp index로 정렬
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

## 규모(3~4달)에 비해 측정단위(1분)이 너무 작기 때문에 다운샘플링으로 추세를 파악해보려함
df = df[:-1]  # machine_status 열을 제외함
df_h_m = df.resample("h").mean(numeric_only=True)  # 시간 단위로 리샘플링 후 평균
# 시간대 구분 — 24시간 안의 운영 패턴
df_D_m = df.resample("D").mean(numeric_only=True)  # 일 단위
# 흐름 부각 — 일별 평균이 추세를 또렷하게
df_W_m = df.resample("W").mean(numeric_only=True)  # 주 단위
# 큰 그림 — 주 단위 평균으로 장기 변화 관찰

## 이동평균 · 이동표준편차 · 변화율 · 시차 변수 중 3종 이상 생성  (시계열 담당 과제 의무)
print(df_W_m)

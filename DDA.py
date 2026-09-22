import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/sensor.csv")
# 1. 앞부분 확인
print(df.head())
# 2. 데이터 크기
print("\nshape")
print(df.shape)
# 220320,54(55 정답 레이블 빼면~)
# 3. 컬럼명
print("\ncolumns")
# timestamp 빼곤 sensor_00~51
print(df.columns.tolist())
# 4. 자료형 / 결측 여부
print("\ninfo")
df.info()
# 타임스탬프만 str 나머진 모두 실수형이라 따로 변환 필요가 없을 것으로 보임
# 총 220320개 있는데 결측치가 꽤 있음, 15는 통으로 결측 제거해야함
# 5. 기술통계
print("\ndescribe")
print(df.describe().T)
# min 값이 0인 것이 있음 측정이 안된게 의심됨!!,
# 산분위수를 살짝 보니 센서4번처럼 이상치가 존재하는듯!
# 값의 범위 차이가 꽤 있음 비교할때 표준화가 필요할수도!

sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

zero_ratio = (df[sensor_cols] == 0).mean() * 100
print(zero_ratio.sort_values(ascending=False))

# 문자열 데이터 타임스탬프로 바꾸기
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 센서별 0 비율 확인! 7퍼짜리가 ㅣㅆ다? 시간대별로 그러는지 확인해봐야할듯~
cols = [
    "sensor_19",
    "sensor_13",
    "sensor_18",
    "sensor_17",
    "sensor_22",
    "sensor_37",
    "sensor_25",
    "sensor_12",
    "sensor_24",
]
# print("0 비율 재확인!")
# for col in cols:
#     print(col)
#     print(df.loc[df[col] == 0, ["timestamp", col]].head(120))
# 센서 19와 17은 계속 연속적인 시간동안그럼 19는 20분? 17은 2시간이상 나머지는 띄엄띄엄
for col in cols:
    is_zero = df[col].eq(0)

    group = (is_zero != is_zero.shift()).cumsum()

    zero_runs = (
        df[is_zero]
        .groupby(group[is_zero])
        .agg(
            start=("timestamp", "first"), end=("timestamp", "last"), count=(col, "size")
        )
    )

    zero_runs["duration_min"] = (
        zero_runs["end"] - zero_runs["start"]
    ).dt.total_seconds() / 60 + 1

    print("\n", col)
    print(zero_runs.sort_values("duration_min", ascending=False).head(10))
# 0인 데이터 그려보기 ㅠㅠ 안그려짐
# 0이 함께 몇시간동안 나오는 센서들이 있음!!! 점검시간이 유력함~

# for col in cols:
#     plt.figure(figsize=(12, 4))
#     plt.plot(df["timestamp"], df[col])
#     plt.title(col)
#     plt.xlabel("time")
#     plt.ylabel("value")
#     plt.show()

# 6. 결측치
print("\nmissing")
print(df.isnull().sum().sort_values(ascending=False))
# 15번 컬럼은 통으로 결측, 50,51,00도 꽤 큼 결측이 있는 행을 통째로 날리는 건 치명적일 것 같음!
# 결측을 다른값으로 채우는 게 좋을 것 같음
# 7. 중복
print("\nduplicated")
print(df.duplicated().sum())
# 중복되는 값이 없다!
# 8. 유일값 개수
print("\nnunique")
print(df.nunique().sort_values())
# 유일값의 개수가 많다.  센서 7,9,3은 중복값의 범위가 적다. 범주형 값일 확률이!?
time_col = "timestamp"
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col)
print("\n시간 범위")
print(df[time_col].min())
print(df[time_col].max())
# 4월부터 5개월간의 데이터이다.
print("\n시간 간격")
print(df[time_col].diff().value_counts().head(10))
# 1분
print("\n시간 중복")
print(df[time_col].duplicated().sum())
# 중복측정 없었다!
time_diff = df[time_col].diff()

print(time_diff.describe())

print("\n1분이 아닌 간격")
print(time_diff[time_diff != pd.Timedelta(minutes=1)].value_counts().sort_index())
# 모든값이 1분!

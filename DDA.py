import pandas as pd

df = pd.read_csv("sensor.csv")

print(df.head())


print(df.shape)
# 10000 , 14의 크기를 갖고있는 데이터 셋이다.

print("나는 머지~~~")

df.info()
# 결측치가 존재하지 않음, 제품 iD와 타입(나중에 인코딩 해줘도 좋다~) 뺴고 모두 숫자형으로 잘 들어가 있다!
print(df.describe().T)
# Rotational speed 이 친구의 최대값이 평균, 중앙값에 비해 꽤 높다!??
print(df.isnull().sum())
# 결측치 없다.
print(df.duplicated().sum())
# 중복되는 행 없다

# 테스트 주석


# print(df["Machine failure"].value_counts())
# # Machine failure 분포는 0 : 9661 , 1(고장) 339
# print(df["Machine failure"].value_counts(normalize=True))
# # 정상(0) 96.61%, 고장(1) 3.39%로 클래스 불균형이 매우 크다.
# # 따라서 Accuracy만으로 모델 성능을 평가하기 어렵고,
# # Recall, Precision, F1-score 등을 함께 확인할 필요가 있음

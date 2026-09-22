import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay,f1_score,classification_report
from sklearn.dummy import DummyClassifier
def set_korean_font():
    font_candidates = [
        "Malgun Gothic",          # Windows
        "AppleGothic",            # macOS
        "Apple SD Gothic Neo",    # macOS
        "NanumGothic",            # Linux / 설치된 경우
        "Noto Sans CJK KR",       # Linux / 설치된 경우
        "Noto Sans KR",           # 설치된 경우
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

df = pd.read_csv('data/sensor.csv')
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
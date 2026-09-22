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
print(df.shape)
print(df.isna().sum().value_counts())
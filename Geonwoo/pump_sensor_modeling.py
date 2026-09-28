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
# ===================================================================
df = pd.read_csv("data/sensor.csv")

# 불필요한 인덱스 컬럼 제거
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# timestamp 변환 및 정렬
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 센서 컬럼 추출 및 전체 결측 센서 제거
sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

all_missing_cols = [col for col in sensor_cols if df[col].isna().all()]

df = df.drop(columns=all_missing_cols)
sensor_cols = [col for col in sensor_cols if col not in all_missing_cols]

# 센서를 연속형과 이산형 후보로 분류
nunique = df[sensor_cols].nunique()
discrete_cols = nunique[nunique <= 20].index.tolist()
continuous_cols = [col for col in sensor_cols if col not in discrete_cols]

# 시간 기준 보간을 위해 timestamp를 인덱스로 설정
df = df.set_index("timestamp")

# 연속형 센서: 시간 기준, 최대 5개 행까지 보간
if continuous_cols:
    df[continuous_cols] = df[continuous_cols].interpolate(
        method="time",
        limit=5,
        limit_direction="both",
    )

# 이산형 후보 센서: 직전 값으로 최대 5개 행까지 채움
if discrete_cols:
    df[discrete_cols] = df[discrete_cols].ffill(limit=5)

# timestamp를 컬럼으로 복구
df = df.reset_index()

# machine_status는 보간하거나 수정하지 않음
print("결측 행도 유지한 최종 크기:", df.shape)
print("센서별 남은 결측 수:")
print(df[sensor_cols].isna().sum().sort_values(ascending=False).head(10))
print(df.shape)

# 센서 컬럼만
sensor_cols = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[sensor_cols]
normal_mask = df["machine_status"].eq("NORMAL")

model = IsolationForest(
    contamination= 0.01,
    random_state= 42
)

model.fit(X.loc[normal_mask])
pred = model.predict(X)
pred_anomaly = pred == -1

# 방식 1: BROKEN만 이상, RECOVERING 제외
eval_mask_1 = df["machine_status"].isin(["NORMAL", "BROKEN"])
y_true_1 = df.loc[eval_mask_1, "machine_status"].eq("BROKEN")
y_pred_1 = pred_anomaly[eval_mask_1]

# 방식 2: RECOVERING과 BROKEN 모두 이상
y_true_2 = df["machine_status"].isin(["RECOVERING", "BROKEN"])
y_pred_2 = pred_anomaly

# # 예측 결과를 df의 인덱스에 맞춘 Series로 변환
pred_anomaly = pd.Series(pred == -1, index=df.index)

# 혼동행렬 계산: 행은 실제, 열은 예측
cm_1 = confusion_matrix(
    y_true_1.astype(int),
    y_pred_1.astype(int),
    labels=[0, 1],
)

cm_2 = confusion_matrix(
    y_true_2.astype(int),
    y_pred_2.astype(int),
    labels=[0, 1],
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(
    cm_1,
    display_labels=["정상", "이상(BROKEN)"],
).plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
axes[0].set_title("RECOVERING 제외")

ConfusionMatrixDisplay(
    cm_2,
    display_labels=["정상", "이상(RECOVERING/BROKEN)"],
).plot(ax=axes[1], cmap="Oranges", values_format="d", colorbar=False)
axes[1].set_title("RECOVERING을 이상으로 포함")

plt.tight_layout()
# plt.show()

cases = [
    ("RECOVERING 제외", y_true_1, y_pred_1),
    ("RECOVERING을 이상으로 포함", y_true_2, y_pred_2),
]

rows = []

for name, y_true, y_pred in cases:
    tn, fp, fn, tp = confusion_matrix(
        y_true.astype(int),
        y_pred.astype(int),
        labels=[0, 1],
    ).ravel()

    rows.append({
        "평가 기준": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "오탐(FP)": fp,
        "미탐(FN)": fn,
        "탐지(TP)": tp,
        "정상 판정(TN)": tn,
    })

evaluation_table = pd.DataFrame(rows)

print(evaluation_table.round(4).to_string(index=False))
# 제조 센서 데이터 기반 비지도 이상탐지

## 1. 프로젝트 개요
다변량 설비 센서 시계열 데이터를 활용하여
Z-score, Isolation Forest, LSTM Autoencoder 기반 이상탐지를 수행했습니다.

실제 상태값(machine_status)은 모델 학습에는 사용하지 않고,
최종 성능 검증 단계에서만 사용했습니다.

---

## 2. 프로젝트 기간
- 2일

---

## 3. 팀 구성
- 팀장
- 전처리
- 시계열 분석
- 모델링
- 기록

---

## 4. 데이터
- 시계열 센서 데이터
- 약 22만 행
- 익명화 센서 변수
- 상태값: NORMAL / BROKEN / RECOVERING

---

## 5. 전처리
- timestamp 기준 정렬
- 전체 결측 센서 제거
- 0값 유지
- 짧은 결측 구간 보간
- 긴 결측 구간 제거
- timestamp gap 기준 LSTM sequence 분리
- 시간순 Train/Test 70:30 분할
- Train 기준 StandardScaler 적용

---

## 6. 사용 모델

### Z-score
- 각 시점의 최대 절대 Z-score 사용
- Threshold = 3

### Isolation Forest
- 비지도 이상탐지
- contamination = 0.01

### LSTM Autoencoder
- TIME_STEPS = 30
- Reconstruction Error 기반 이상점수
- 95 / 97 / 99 percentile 비교

### Ensemble
- Isolation Forest + LSTM
- 두 모델이 모두 이상으로 판단한 경우 이상 처리

---

## 7. 주요 결과

BROKEN + RECOVERING을 비정상으로 정의한 결과:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Z-score | 0.0028 | 0.9737 | 0.0055 |
| Isolation Forest | 0.7778 | 0.6447 | 0.7050 |
| LSTM p95 | 0.0040 | 0.9474 | 0.0079 |
| LSTM p97 | 0.0111 | 0.8816 | 0.0220 |
| LSTM p99 | 0.1706 | 0.6711 | 0.2720 |
| IF + LSTM p99 | 1.0000 | 0.5921 | 0.7438 |

Isolation Forest는 오탐과 미탐 사이에서 비교적 균형적인 성능을 보였습니다.

---

## 8. 주요 해석
- Z-score는 이상을 많이 탐지했지만 오경보가 매우 많았습니다.
- LSTM은 threshold가 낮을수록 Recall은 높았지만 FP가 크게 증가했습니다.
- IF + LSTM p99는 FP가 0이었지만 미탐이 증가했습니다.
- Isolation Forest는 Precision 0.7778, Recall 0.6447로 비교적 균형적인 성능을 보였습니다.
- 실제 고장 직전 LSTM Reconstruction Error가 점진적으로 증가하는 패턴을 확인했습니다.

---

## 9. 한계 및 개선 방향
- 실제 BROKEN 라벨이 매우 적음
- 고정 Threshold 방식의 한계
- Rolling Mean / Std / 변화율 Feature 추가 필요
- Event 기반 이상탐지 평가 필요
- 고장 전 Lead Time 분석 필요

---

## 10. 프로젝트 구조

```text
project/
├─ data/
├─ models/
├─ results/
├─ preprocessing.py
├─ PDA.py
└─ README.md
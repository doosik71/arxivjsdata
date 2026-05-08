# VGTL-net: Amplitude-Independent Machine Learning for PPG through Visibility Graphs and Transfer Learning

Yuyang Miao, Harry J. Davies, Danilo P. Mandic (2024)

## 🧩 Problem to Solve

본 연구는 광전용적맥파(Photoplethysmography, PPG) 신호를 이용한 생체 지표 추출 및 상태 예측 과정에서 발생하는 기존 방법론들의 한계를 해결하고자 한다. PPG 신호는 웨어러블 기기를 통해 심박수, 혈압, 혈관 노화 등 다양한 생체 정보를 제공하지만, 기존의 딥러닝 및 신호 처리 알고리즘들은 다음과 같은 문제점을 가지고 있다.

첫째, 수동으로 설계된 규칙에 기반한 과도한 전처리가 필요하며, 이는 인간의 캘리브레이션에 크게 의존한다. 둘째, 신호의 진폭(Amplitude) 정보를 직접 입력 특성으로 사용하기 때문에 연령, 피부 두께, 피부 톤과 같은 개인적 특성에 의해 결과가 크게 영향을 받는다. 셋째, 시계열 데이터의 특성상 아핀 변환(Affine Transformation)이나 신호 품질 저하에 매우 민감하여, 서로 다른 데이터셋 간의 일반화 성능이 떨어진다.

따라서 본 논문의 목표는 진폭에 독립적이며 아핀 변환에 불변하는(Invariant) 분석 프레임워크를 구축하여, 최소한의 전처리만으로도 다양한 태스크와 데이터셋에서 강건한 일반화 성능을 보이는 PPG 신호 처리 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **시계열 PPG 신호를 그래프 이론과 컴퓨터 비전 알고리즘을 결합하여 이미지 형태로 변환한 뒤, 전이 학습(Transfer Learning)을 적용하는 것**이다.

구체적으로는 **Visibility Graph (VG)** 기법을 사용하여 1차원 시계열 신호를 복잡한 그래프 네트워크로 변환한다. 이 과정에서 신호의 절대적인 진폭 정보는 버리고 기하학적 구조 정보만을 보존함으로써 진폭 독립성을 확보한다. 이후 이 그래프의 **인접 행렬(Adjacency Matrix)**을 2차원 이미지로 간주하여, ImageNet으로 사전 학습된 고성능 컴퓨터 비전 모델(CNN)에 입력함으로써 1차원 신호 분석 문제를 2차원 이미지 인식 문제로 치환하여 해결한다.

## 📎 Related Works

기존의 PPG 분석 연구들은 주로 다음과 같은 접근 방식을 취했다.

- **호흡률 추출:** Empirical Mode Decomposition (EMD)을 통해 신호를 분해하거나, 특정 물리적/통계적 특성을 추출하여 신경망에 입력하는 방식이 사용되었다.
- **혈관 노화 예측:** Hilbert 변환을 통한 복조(Demodulation) 및 엔벨로프(Envelope) 추출과 같은 복잡한 전처리를 거친 후 ResNet이나 SVM을 적용하였다.
- **혈압 예측:** 맥파 전달 시간(Pulse Transit Time, PTT)이나 Dicrotic Notch와 같은 수동 설계된 생리학적 특성을 추출하여 AdaBoost나 MLP-Mixer 등의 모델로 예측하였다.

이러한 기존 방식들은 신호의 진폭에 의존하며, 전처리 과정에서 많은 파라미터 선택이 필요하고, 1차원 신호를 처리하기 위해 컴퓨터 비전 모델을 무리하게 수정하여 적용하는 등 구조적인 한계가 존재한다.

## 🛠️ Methodology

### 1. Visibility Graph (VG) 변환

시계열 데이터의 각 샘플 $y_i$를 그래프의 정점(Vertex)으로 간주하고, 특정 조건을 만족하는 두 정점 사이에 에지(Edge)를 연결한다. 두 샘플 $y_a$와 $y_b$가 서로 '보인다'고 판단하여 에지를 연결하는 조건은 다음과 같다.

$$y_c < y_b + (y_a - y_b) \frac{t_b - t_c}{t_b - t_a}$$

여기서 $t_a, t_b, t_c$는 각각 샘플의 시간 인덱스이며, $a$와 $b$ 사이의 모든 샘플 $y_c$가 이 식을 만족해야 한다. 즉, 두 점을 잇는 직선이 중간에 다른 점에 의해 가려지지 않아야 연결되는 구조이다. 이 방식은 신호의 절대적인 값보다 상대적인 구조를 반영하므로, 수직/수평 이동, 스케일링, 선형 추세 추가와 같은 **아핀 변환(Affine Transformation)에 불변**하는 특성을 가진다.

### 2. Graph to Image: Adjacency Matrix

생성된 Visibility Graph는 인접 행렬 $A$로 변환된다. 가중치가 없는 그래프의 경우, 연결되면 1, 아니면 0으로 표시되는 $N \times N$ 크기의 이진 행렬이 생성된다. 이 행렬은 흑백 이미지로 간주될 수 있으며, 표준 2D CNN 모델의 입력으로 직접 사용할 수 있다.

### 3. RGB 채널 융합 (Information Fusion)

본 논문은 RGB 3채널을 활용하여 서로 다른 정보를 통합적으로 입력한다.

- **혈압 예측 태스크:**
  - Red 채널: PPG 신호의 VG 인접 행렬
  - Green 채널: ECG 신호의 VG 인접 행렬 (PPG와 ECG 간의 위상 차이를 통해 PTT 정보 반영)
  - Blue 채널: PPG 신호의 1차 미분값의 VG 인접 행렬
- **혈관 노화 예측 태스크:**
  - Red 채널: PPG 신호의 VG 인접 행렬
  - Green 채널: 진폭 반전(Amplitude-inverted) PPG 신호의 VG 인접 행렬
  - Blue 채널: 진폭 반전 PPG 신호의 경사 가중치(Slope-weighted) VG 인접 행렬

### 4. VGTL-net 학습 절차

전체 파이프라인은 다음과 같다.

1. PPG 신호를 고정 길이 윈도우 또는 고정된 펄스 수로 분할한다.
2. 각 분할된 신호를 VG를 통해 인접 행렬 이미지로 변환한다.
3. RGB 채널에 위 정보를 할당하여 사전 학습된 모델(VGG19, CoAtNet, ConvNeXt v2)에 입력한다.
4. 모델의 분류 레이어를 제거하고, 태스크에 맞는 MLP(Multi-Layer Perceptron) 층을 추가하여 회귀(Regression) 또는 분류(Classification)를 수행한다.

## 📊 Results

### 1. 혈압 추정 (Blood Pressure Estimation)

- **데이터셋:** MIMIC II (UCI Machine Learning Repository).
- **평가 지표:** 평균 절대 오차(MAE), 평균 오차(ME), 피어슨 상관계수($\rho$).
- **결과:** CoAtNet과 ConvNeXt v2를 결합(Concatenate)한 모델이 가장 우수한 성능을 보였다.
  - SBP MAE: $3.11 \pm 3.92 \text{ mmHg}$
  - DBP MAE: $1.94 \pm 2.37 \text{ mmHg}$
  - 이는 기존의 PPG+ECG 기반 최신 모델들보다 정량적으로 더 우수한 수치이다.

### 2. 혈관 노화 예측 (Vascular Ageing)

- **데이터셋:** Mendeley의 Real-World PPG Dataset (35명 대상).
- **태스크:** 정확한 나이 예측(회귀) 및 4개 연령대 분류(분류).
- **결과:**
  - **분류:** 4개 클래스 분류 정확도 $97.20 \pm 0.23\%$를 달성하여, 기존 Dall’Olio et al. (87.14%) 대비 월등한 성능을 보였다. 특히 수렴 속도가 훨씬 빨랐다.
  - **회귀:** 평균 MAE $1.41$년, 표준편차 $0.14$년의 매우 높은 정밀도로 나이를 예측하였다.

## 🧠 Insights & Discussion

### 강점

VGTL-net은 시계열 데이터를 그래프-이미지로 변환함으로써, 1차원 데이터 분석의 한계를 극복하고 현대 컴퓨터 비전의 강력한 사전 학습 모델(Pre-trained Models)을 그대로 활용할 수 있게 했다. 특히 Visibility Graph의 수학적 특성 덕분에 PPG 신호의 고질적인 문제인 베이스라인 완더링(Baseline Wandering)이나 진폭 변화에 매우 강건하다는 점이 입증되었다. 또한, 복잡한 필터링이나 수동 특성 추출 없이 단순한 변환만으로 SOTA 성능을 낸 점이 고무적이다.

### 한계 및 비판적 해석

본 연구는 소규모 데이터셋(특히 혈관 노화 데이터셋)에서 높은 성능을 보였으나, 실제 임상 환경의 훨씬 더 방대하고 노이즈가 심한 데이터셋에서도 동일한 일반화 능력을 유지할지는 추가 검증이 필요하다. 또한, VG 생성 과정의 시간 복잡도가 신호의 길이 $N$에 대해 $O(N^2)$ 수준이므로, 매우 긴 신호를 실시간으로 처리할 때 계산 효율성 문제가 발생할 수 있다.

## 📌 TL;DR

본 논문은 PPG 신호를 **Visibility Graph $\rightarrow$ 인접 행렬(이미지)** 순으로 변환하여, 진폭에 독립적이고 아핀 변환에 불변하는 분석 프레임워크인 **VGTL-net**을 제안하였다. 이 방식은 1차원 생체 신호를 2차원 이미지로 치환함으로써 ImageNet으로 학습된 CV 모델의 전이 학습 능력을 극대화하였으며, 혈압 추정 및 혈관 노화 예측 태스크에서 기존 방식보다 적은 전처리만으로도 SOTA 성능을 달성하였다. 이 연구는 향후 다양한 생체 신호 분석을 위한 보편적인 프레임워크로 확장될 가능성이 높다.

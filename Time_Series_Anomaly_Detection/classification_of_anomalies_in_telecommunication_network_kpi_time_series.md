# Classification of Anomalies in Telecommunication Network KPI Time Series

Korantin Bordeau–Aubert, Justin Whatley, Sylvain Nadeau, Tristan Glatard, Brigitte Jaumard (2023)

## 🧩 Problem to Solve

통신 네트워크의 복잡성과 규모가 증가함에 따라 네트워크 성능 지표(Key Performance Indicators, KPI) 시계열 데이터에서 이상치를 자동으로 탐지하는 시스템에 대한 관심이 높아졌다. 그러나 기존 연구들은 대부분 이상치 탐지(Anomaly Detection)에 집중되어 있으며, 탐지된 이상치가 구체적으로 어떤 유형인지 분류(Classification)하는 연구는 상대적으로 부족한 실정이다.

이상치 분류가 제대로 이루어지지 않으면 네트워크 관리자는 이상치의 특성을 파악하고 적절한 대응책을 마련하는 데 어려움을 겪게 된다. 따라서 본 논문의 목표는 시뮬레이션된 데이터와 실제 네트워크 KPI 데이터를 모두 활용하여 이상치를 효과적으로 탐지하고 분류할 수 있는 모듈형 이상치 분류 프레임워크(Modular Anomaly Classification Framework)를 제안하고 그 성능을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이상치 탐지기(Detector)와 분류기(Classifier)를 분리하여 독립적인 모듈로 설계하는 것이다. 이러한 모듈형 구조를 통해 탐지와 분류 작업을 각각 최적화하여 처리할 수 있다. 주요 기여 사항은 다음과 같다.

1. **KPI 시계열 시뮬레이터 개발**: 실제 네트워크 KPI의 동작(계절성, 추세, 노이즈 등)을 모방하여 합성 데이터를 생성하는 시뮬레이터를 구축하여, 학습 데이터 부족 문제를 해결하고 다양한 이상치 시나리오를 생성하였다.
2. **TCN 기반의 탐지 및 분류 모델 제안**: Temporal Convolutional Network(TCN)를 활용하여 시계열의 시간적 의존성을 효과적으로 캡처함으로써 이상치 탐지와 분류 성능을 향상시켰다.
3. **Sim-to-Real 전이 가능성 검증**: 시뮬레이션 데이터(SIM)로 학습된 분류 모델을 실제 데이터(REAL)에 적용하여, 합성 데이터 기반의 학습이 실제 환경에서도 유효함을 입증하였다.

## 📎 Related Works

논문에서는 시계열 이상치의 정의와 탐지 및 분류 방법론에 대해 다룬다.

- **이상치의 정의**: Cho et al.은 이상치를 Point, Contextual, Collective의 세 가지 유형으로 구분하였으며, Foorthuis는 데이터 타입, 관계의 카디널리티, 이상치 레벨 등 5가지 차원을 통해 이상치를 정의하였다. 본 연구는 특히 정량적이며 단변량인 시계열 데이터에서 Single point, Temporary change, Level shift, Variation change의 4가지 유형에 집중한다.
- **이상치 탐지**: 전통적인 ARIMA 모델부터 RNN, CNN, LSTM-AD와 같은 딥러닝 기반 예측 모델이 사용되었다. 특히 TCN은 RNN/CNN의 장점을 결합하여 병렬 연산이 가능하고 긴 의존성을 잘 캡처한다는 점에서 우수성이 입증되었다.
- **시계열 분류(TSC)**: k-Nearest Neighbors(kNN)와 Dynamic Time Warping(DTW)의 조합, 그리고 Random Forest의 발전 형태인 Supervised Time Series Forest(STSF) 등이 제안되었다. 최근에는 CNN 기반의 InceptionTime 등이 높은 성능을 보이고 있다.

본 연구는 이러한 기존 기법들을 통합하여 네트워크 KPI라는 특정 도메인에 맞춘 탐지-분류 파이프라인을 구축했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조

제안하는 프레임워크는 **[시뮬레이션 $\rightarrow$ 탐지 $\rightarrow$ 데이터 전처리 $\rightarrow$ 분류]**의 순서로 구성된다. 탐지기가 이상 지점을 식별하면, 해당 지점을 중심으로 분석 윈도우(Analysis Window)를 생성하고, 이를 분류기가 입력받아 유형을 결정한다.

### 2. 이상치 시뮬레이션 (Anomaly Simulation)

실제 네트워크 지연 시간(Latency)을 모방하기 위해 일간, 주간, 월간 계절성을 포함한 기저 신호를 생성한다.

- **기저 신호 생성**:
    $$X(t) = \prod_{s=0}^2 (A_s \times \sin(\frac{2\pi}{T_s}t) + \mu_s)$$
    여기서 $A_s$는 진폭, $T_s$는 주기(분 단위), $\mu_s$는 평균을 의미한다.
- **이상치 주입**: 정의된 4가지 유형(Single point, Temporary change, Level shift, Variation change)을 특정 강도 $\alpha$와 길이 $\lambda$를 가지고 주입한다.
- **노이즈 주입**: Gaussian multiplicative white noise를 추가하여 실제 데이터와 유사한 변동성을 부여한다.

### 3. 이상치 탐지 (Anomaly Detection)

TCN 모델을 사용하여 시계열 예측을 수행하고, 예측값과 실제값의 차이를 통해 이상치를 탐지한다.

- **학습 및 손실 함수**: Adam 옵티마이저를 사용하며, Gaussian likelihood function 기반의 $\beta$-Negative log likelihood loss를 통해 학습한다.
- **탐지 기준**: 예측된 시계열 $\check{X}$와 실제값 $\hat{X}$ 사이의 차이가 95% 신뢰 구간 $\delta$를 벗어날 때 이상치로 판정한다.
    $$|\hat{X} - \check{X}| > \delta$$
- **분석 윈도우**: 탐지된 이상 지점을 중심으로 전후 $m$ 기간을 포함하는 크기 $2m$의 분석 윈도우를 생성한다.

### 4. 이상치 분류 (Anomaly Classification)

분류기는 원본 신호에서 계절성과 추세를 제거한 잔차(Residual) 성분만을 입력으로 사용한다.

- **시계열 분해 (Decomposition)**: Moving average 분해법을 사용하여 시계열을 다음과 같이 분리한다.
    $$\hat{X} = T_i + U_i + R_i$$
    ($T_i$: 추세, $U_i$: 계절성, $R_i$: 잔차). 분류기는 $R_i$를 사용하여 학습 및 추론을 수행한다.
- **사용 모델**:
  - **kNN**: DTW 등 다양한 거리 측정 방식을 적용한 표준 분류기.
  - **STSF**: 인터벌 기반의 의사결정 트리 앙상블 모델.
  - **TCN**: Softmax 활성화 함수와 Sparse categorical cross-entropy loss를 사용하는 딥러닝 모델.

## 📊 Results

### 1. 실험 설정

- **데이터셋**:
  - `REAL`: 65개의 실제 네트워크 지연 시간 시계열 데이터.
  - `aREAL`: `REAL` 데이터에서 탐지 모델을 통해 추출하고 수동으로 라벨링한 분석 윈도우.
  - `SIM` & `aSIM`: 시뮬레이터를 통해 생성한 합성 데이터 및 그로부터 추출된 분석 윈도우.
- **평가 지표**: Micro F1-score 및 혼동 행렬(Confusion Matrix).

### 2. 주요 결과

- **SIM-SIM 실험**:
  - 탐지 모델은 노이즈 수준이 높아질수록 F1-score가 감소하는 경향을 보였다.
  - 분류 모델 중 TCN과 STSF가 가장 우수한 성능을 보였으며, 전체 평균 F1-score는 불균형 데이터셋에서 0.7, 균형 데이터셋에서 0.73을 기록하였다.
- **SIM-REAL 실험 (Sim-to-Real)**:
  - 시뮬레이션 데이터로만 학습한 TCN과 STSF 모델이 실제 데이터의 Single point 및 Temporary change 이상치에 대해 높은 F1-score(>0.6)를 기록하였다.
  - 다만, Level shift와 Variation change의 경우 실제 데이터에서 발생 빈도가 매우 낮아 분류 성능이 상대적으로 낮게 측정되었다. 전체 F1-score는 불균형 데이터셋 기준 0.39로 감소하였으나, 이는 클래스 불균형과 희소성 문제에 기인한다.

## 🧠 Insights & Discussion

### 강점

- **합성 데이터의 유효성**: 실제 네트워크 데이터는 라벨링된 이상치 샘플을 확보하기 매우 어렵다. 본 연구는 정교한 시뮬레이터를 통해 생성한 데이터로 모델을 학습시켜도 실제 데이터의 주요 이상치 유형을 효과적으로 분류할 수 있음을 보여주었다.
- **모듈형 구조의 유연성**: 탐지와 분류를 분리함으로써, 탐지 단계에서 발생한 오차가 분류 단계로 전이되는 것을 분석 윈도우와 잔차 분해(Decomposition)를 통해 완화하였다.

### 한계 및 비판적 해석

- **윈도우 크기의 고정**: 현재 2시간(120분)으로 고정된 분석 윈도우 크기를 사용하고 있다. 이는 Level shift와 같이 긴 호흡의 이상치를 분류할 때 변별력을 떨어뜨리는 원인이 된다. 논문에서도 언급되었듯 적응형(Adaptive) 윈도우 크기 도입이 필요하다.
- **단순화된 시뮬레이션**: 실제 환경에서는 여러 종류의 이상치가 동시에 발생하는 '중첩 이상치(Superimposed anomalies)'가 빈번하지만, 현재 시뮬레이터는 단일 이상치 주입 방식만을 지원한다.
- **노이즈 민감도**: 노이즈 레벨이 증가함에 따라 성능이 급격히 저하되는 모습이 관찰되었다. 분류 전 단계에서 더 강력한 전처리 필터링 기법이 적용되어야 할 필요가 있다.

## 📌 TL;DR

본 논문은 통신 네트워크 KPI 시계열 데이터의 이상치를 탐지하고 유형별로 분류하는 모듈형 프레임워크를 제안한다. TCN 기반의 예측 모델로 이상치를 탐지하고, 시뮬레이터로 생성한 합성 데이터를 통해 학습된 TCN 및 STSF 분류기로 이상치 유형을 결정한다. 실험 결과, 합성 데이터로 학습된 모델이 실제 네트워크의 단순 이상치(Single point, Temporary change)를 효과적으로 분류할 수 있음을 입증하여, 실제 데이터 부족 문제를 해결할 수 있는 실용적인 방향성을 제시하였다.

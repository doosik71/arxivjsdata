# Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction

Chih-Yu Lai, Fan-Keng Sun, Zhengqi Gao, Jeffrey H. Lang, Duane S. Boning (2023)

## 🧩 Problem to Solve

시계열 이상치 탐지(Time Series Anomaly Detection)는 데이터 내에 존재하는 패턴의 복잡성과 다양성으로 인해 매우 어려운 과제이다. 특히 본 논문은 **Point Anomaly(점 이상치)**와 **Contextual Anomaly(문맥 이상치)**를 동시에 효과적으로 탐지하는 것 사이의 트레이드오프(Trade-off) 문제를 해결하고자 한다.

Point Anomaly는 단일 시점의 데이터만으로도 식별 가능한 이상치인 반면, Contextual Anomaly는 시계열의 문맥적 정보, 즉 시간적 의존 관계를 파악해야만 식별할 수 있다. 기존의 Sequence-based 모델들은 문맥 이상치를 잡기 위해 시간적 관계를 학습하지만, 이 과정에서 모델의 복잡도가 증가하고 재구성 결과에 노이즈가 발생하여 오히려 Point Anomaly에 대한 정밀도가 떨어지는 문제가 발생한다. 반대로 Point-based 모델은 시간적 정보를 무시하므로 문맥 이상치를 탐지할 수 없다. 따라서 본 연구의 목표는 두 모델의 장점을 통합하여 Point Anomaly의 정밀도를 유지하면서도 Contextual Anomaly를 효과적으로 탐지할 수 있는 비지도 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Nominality Score(정상성 점수)**라는 개념을 도입하여, Point-based 재구성 모델과 Sequence-based 재구성 모델의 출력을 결합하는 것이다.

단순히 두 모델의 오차를 합치는 것이 아니라, 데이터가 얼마나 '정상적인지'를 나타내는 Nominality Score를 계산하고, 이를 게이트(Gate)로 활용하여 원래의 Anomaly Score를 조건화(Conditioning)한 **Induced Anomaly Score**를 유도한다. 이를 통해 정상 구간에서는 이상치 점수의 전파를 억제하고, 이상 구간에서는 점수를 증폭시킴으로써 탐지 성능을 높이는 이론적 근거를 제시하고 이를 실험적으로 증명하였다.

## 📎 Related Works

기존의 재구성 기반(Reconstruction-based) 기법들은 주로 네트워크 아키텍처의 개선(예: LSTM-VAE, MAD-GAN, TranAD 등)이나, 단순한 재구성 오차 대신 재구성 확률(Reconstruction Probability)과 같은 정교한 Anomaly Score를 설계하는 방향으로 발전해 왔다.

그러나 기존 방식들은 대부분 단일 모델(주로 Sequence-based)에 의존하며, 앞서 언급한 Point-Contextual 탐지 트레이드오프 문제를 명시적으로 다루지 않았다. 본 논문은 아키텍처의 복잡성을 높이는 대신, Point-based와 Sequence-based 모델의 오차를 통합하는 새로운 점수 산출 메커니즘을 제안함으로써 기존 접근 방식과 차별점을 둔다.

## 🛠️ Methodology

### 1. 문제 정의 및 이단계 편차(Two-stage Deviation)

논문은 관측된 시계열 데이터 $x^0_t$가 기저의 정상 시계열 $x^*_t$로부터 두 단계의 편차를 거쳐 생성되었다고 가정한다.
$$x^0_t = x^*_t + \Delta x^c_t + \Delta x^p_t$$
여기서 $\Delta x^c_t$는 **In-distribution deviation**으로 문맥 이상치와 관련이 있으며, $\Delta x^p_t$는 **Out-of-distribution deviation**으로 점 이상치와 관련이 있다. 즉, $x^c_t = x^*_t + \Delta x^c_t$는 여전히 정상 데이터 집합 내에 존재하지만 시간적 흐름상 잘못된 값이며, $x^0_t$는 이조차 벗어난 값이다.

### 2. Nominality Score $N(t)$

Nominality Score는 해당 시점의 데이터가 얼마나 정상적인지를 수치화한 것이다. 본 논문에서는 점 기반 재구성 결과 $\hat{x}_{c,t}$와 시퀀스 기반 재구성 결과 $\hat{x}_{*,t}$ 사이의 L2-노름 비율로 이를 정의한다.
$$N(t) \triangleq \frac{\|\hat{x}_{c,t} - \hat{x}_{*,t}\|^2_2}{\|x^0_t - \hat{x}_{*,t}\|^2_2}$$
이 값은 Point-based 모델이 점 이상치를 제거하여 정상 범위 내의 값 $\hat{x}_{c,t}$를 잘 찾아냈을 때, $\Delta x^p_t$의 영향력이 클수록 작아지며, 결과적으로 이상치일수록 낮은 값을 가지는 특성을 갖는다.

### 3. Induced Anomaly Score $\hat{A}(t)$

유도된 이상치 점수 $\hat{A}(t)$는 특정 윈도우 $d$ 내의 Anomaly Score $A(\tau)$를 합산하되, 그 사이의 Nominality Score에 의해 제어되는 게이트 함수 $g_{\theta_N}$을 곱하여 계산한다.
$$\hat{A}(t) \triangleq \sum_{\tau=\max(1, t-d)}^{\min(T, t+d)} A(t; \tau)$$
여기서 $A(t; \tau)$는 다음과 같이 정의된다.
$$A(t; \tau) \triangleq A(\tau) \prod_{k=\min(\tau+1, t)}^{\max(t-1, \tau-1)} g_{\theta_N}(N(k))$$
게이트 함수 $g_{\theta_N}(N)$은 $N$이 클수록(정상일수록) 작은 값을 가져 $A(\tau)$의 전파를 차단한다. 논문에서는 두 가지 함수를 제안한다.

- **Soft Gate**: $g_{\theta_N}(N) \triangleq \max(0, 1 - \frac{N}{\theta_N})$
- **Hard Gate**: $g_{\theta_N}(N) \triangleq \mathbb{1}_{N < \theta_N}$

### 4. 모델 아키텍처 ($M_{pt}$ 및 $M_{seq}$)

- **Point-based Model ($M_{pt}$)**: Performer 기반의 Autoencoder를 사용하여 각 시점을 독립적으로 재구성한다. 이는 Point Anomaly를 효과적으로 억제하여 $\hat{x}_{c,t}$를 추정하는 역할을 한다.
- **Sequence-based Model ($M_{seq}$)**: Performer 기반의 Stacked Encoder를 사용한다. 특히 주변 $2\gamma$개의 포인트로부터 중앙의 $\delta$개 포인트를 예측하게 함으로써 모델이 강제적으로 시간적 의존성을 학습하도록 설계되었다. 이를 통해 $\hat{x}_{*,t}$를 추정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: SWaT, WADI, PSM, MSL, SMAP, SMD, trimSyn 등 다양한 도메인의 7개 데이터셋을 사용하였다.
- **평가 지표**: 모든 가능한 임계값($\theta_a$)에 대해 최대 F1 스코어를 산출하는 $F1^*$ (Point-wise F1 score)를 주 지표로 사용하였다. 이는 Point-adjustment를 적용한 F1 스코어가 지나치게 낙관적이라는 최근의 비판을 반영한 것이다.
- **비교 대상**: DAGMM, LSTM-VAE, OmniAnomaly, TranAD 등 최신 딥러닝 기반 모델 및 Simple Heuristics와 비교하였다.

### 2. 주요 결과

- **정량적 성능**: NPSR은 거의 모든 데이터셋에서 SOTA 모델들을 상회하는 성능을 보였다. 특히 WADI, MSL, SMAP, SMD 등의 데이터셋에서 압도적인 $F1^*$를 기록하였다.
- **Ablation Study**: 단순한 Smoothing(Soft/Hard gate 없이 평균 내는 것)보다 Nominality Score 기반의 게이트를 적용했을 때 성능이 향상됨을 확인하였다. 특히 Soft Gate가 Hard Gate보다 일반적으로 더 안정적이고 높은 성능을 보였는데, 이는 실제 데이터에서 정상성과 이상성의 점수 분포가 상당 부분 겹쳐 있기 때문으로 해석된다.
- **Point vs Sequence**: $M_{pt}$만 사용했을 때도 Point Anomaly 탐지에는 매우 강력했으나, $M_{seq}$와 결합하여 $\hat{A}(t)$를 산출했을 때 Contextual Anomaly까지 잡아내며 전체적인 F1 스코어가 상승함을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 이론적 기여

본 논문은 단순한 모델 앙상블이 아니라, 시계열 이상치의 특성을 '이단계 편차'라는 수학적 프레임워크로 정의하고, 이에 기반한 Nominality Score를 통해 이상치 점수를 조건화했다는 점에서 학술적 가치가 높다. 특히 $F1^*$ 지표를 사용하여 Point-adjustment의 함정을 피하고 모델의 실제 변별력을 엄격하게 평가하였다.

### 2. 한계점 및 비판적 해석

- **저차원 데이터의 취약성**: $M_{pt}$는 단일 시점의 분포를 학습하므로, 변수 개수가 매우 적은(특히 단변량) 데이터셋에서는 통계적으로 유의미한 표현을 학습하기 어렵다. 실제로 논문에서도 단변량 데이터인 MGAB 벤치마크에서는 $M_{seq}$가 훨씬 우월함을 언급하며 $M_{pt}$의 한계를 인정하고 있다.
- **하이퍼파라미터 의존성**: 게이트 임계값 $\theta_N$, 유도 길이 $d$, 그리고 $A(\cdot)$를 결정하는 모델 선택 등이 도메인 지식에 의존하며, 이를 자동으로 최적화하는 방법론이 제시되지 않았다.
- **모델 선택의 문제**: 현재는 $A(\cdot)$로 $M_{pt}$의 오차를 기본적으로 사용하지만, 어떤 상황에서는 $M_{seq}$의 오차가 더 유용할 수 있다. 이에 대한 동적 선택 메커니즘이 부재하다는 점이 아쉬운 부분이다.

## 📌 TL;DR

본 논문은 Point Anomaly와 Contextual Anomaly 탐지 간의 트레이드오프를 해결하기 위해, Point-based와 Sequence-based 재구성 모델을 결합한 **NPSR** 프레임워크를 제안한다. 핵심은 두 모델의 재구성 오차 비율로 **Nominality Score**를 정의하고, 이를 게이트로 사용하여 이상치 점수를 전파/억제하는 **Induced Anomaly Score**를 산출하는 것이다. 실험 결과, NPSR은 기존 SOTA 모델들보다 높은 Point-wise F1 스코어를 달성하며 두 종류의 이상치를 모두 효과적으로 탐지할 수 있음을 입증하였다. 이 연구는 고차원 시계열 데이터의 정밀한 이상치 탐지가 필요한 산업 현장(수처리, 서버 모니터링 등)에 유용하게 적용될 가능성이 높다.

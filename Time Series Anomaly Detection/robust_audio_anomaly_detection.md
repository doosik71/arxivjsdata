# ROBUST AUDIO ANOMALY DETECTION

Wo Jae Lee, Karim Helwani, Srikanth Tenneti, & Arvindh Krishnaswamy (2021)

## 🧩 Problem to Solve

본 논문은 노이즈가 포함된 훈련 데이터(noisy training data)를 기반으로, 이전에 본 적 없는 이상 소리(unseen anomalous sounds)를 탐지하기 위한 **이상치 강건한 다변량 시계열 모델(outlier robust multivariate time series model)** 개발을 목표로 한다.

일반적인 이상 탐지 모델은 훈련 데이터에 정상 데이터만 존재한다고 가정하지만, 실제 산업 환경에서 수집되는 진동 센서나 마이크로폰 데이터는 여러 외부 요인으로 인해 오염된 경우가 많다. 기존의 모델들은 다중 타임스텝에 걸친 시간적 의존성을 캡처하면서 동시에 훈련 데이터의 오염(contamination)에 대응하는 능력이 부족하다는 한계가 있다. 따라서 본 연구는 훈련 데이터 내의 이상치에 강건하면서도 다중 해상도(multiple resolutions)에서 시계열 역학을 학습할 수 있는 모델을 제안하여, 오탐(false positive)과 미탐(false negative)을 동시에 줄이고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **이상치에 강건한 확률 밀도 함수(Outlier Robust Probability Density Function)를 사용하여 시계열의 혁신(innovation) 과정을 모델링**하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Student-t Mixture Model (SMM) 도입**: 가우시안 분포보다 꼬리가 두꺼운(heavy-tailed) Student-t 분포를 사용하여 훈련 데이터의 이상치에 덜 민감한 강건한 파라미터 추정을 가능하게 하였다.
2. **다중 해상도 아키텍처(Multi-resolution Architecture)**: 1D-Convolutional 레이어를 통해 다양한 해상도에서 특징을 추출함으로써 전역적 특징(global features)을 효과적으로 포착한다.
3. **Attention 메커니즘 결합**: GRU(Gated Recurrent Unit) 기반의 재귀 신경망에 Attention 메커니즘을 추가하여 긴 시퀀스 데이터 내에서 중요한 시간적 패턴을 더 잘 식별하도록 설계하였다.

## 📎 Related Works

논문에서는 이상 탐지 접근 방식을 크게 생성 모델(Generative)과 판별 모델(Discriminative)로 구분하여 설명한다.

- **생성 모델**: Autoencoder(AE)나 신경망 기반 밀도 추정(Neural Density Estimation) 방식이 포함된다. 그러나 Nalisnick et al. (2019)은 잘 훈련된 생성 모델이 훈련 데이터보다 오히려 분포 외(out-of-distribution) 데이터에 더 높은 확률을 부여할 수 있는 한계가 있음을 지적하였다.
- **판별 모델**: 정상 데이터에 섭동을 가해 합성 이상 데이터를 생성하여 학습하는 방식을 취한다. 하지만 이는 도메인 특성이 강해 적절한 합성 데이터를 생성하는 것이 매우 어렵다는 단점이 있다.
- **기존 시계열 모델**: autoregressive 모델이나 LSTM 기반의 재귀 구조가 사용되어 왔으나, 본 논문은 이러한 구조들이 훈련 데이터의 오염에 취약하다는 점을 지적하며 이를 해결하기 위한 강건한 밀도 추정 방식을 제안한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 모델은 과거의 시계열 데이터 $x_{t-1:0}$가 주어졌을 때 다음 샘플 $x_t$의 조건부 확률 밀도 함수(pdf)를 예측하는 구조이다.
$$P(x_t | x_{t-1}, \dots, 0) \approx P(x_t | h_t)$$
여기서 $h_t$는 신경망을 통해 추출된 은닉 상태(hidden state)이다.

### 2. 강건한 확률 밀도 모델: Student-t Mixture Model (SMM)

본 연구는 가우시안 분포 대신 Student-t 분포의 혼합 모델을 사용한다. Student-t 분포는 자유도(degree of freedom) 파라미터 $\nu$를 통해 이상치에 대한 민감도를 조절할 수 있다. 모델의 파라미터화된 확률 밀도 함수는 다음과 같다.

$$P(x_t | x_{t-1:0}; \theta) = \sum_{i=1}^{c} \alpha_i f(x_t | x_{t-1:0}; \mu_i, \Sigma_i, \nu_i)$$

여기서 $\alpha_i$는 각 성분의 책임도(responsibility), $c$는 성분의 개수이며, $f(\cdot)$는 다음과 같은 Student-t 분포 함수이다.
$$f(y; \mu, \Sigma, \nu) = \frac{\Gamma((\nu+P)/2)}{\Gamma(\nu/2)\nu^{P/2}\pi^{P/2}|\Sigma|^{1/2}} \cdot \left[ 1 + \frac{1}{\nu}(y-\mu)^T \Sigma^{-1} (y-\mu) \right]^{-(\nu+P)/2}$$
($P$는 차원, $\Gamma(\cdot)$는 감마 함수이다.)

### 3. 네트워크 아키텍처 및 학습 절차

- **특징 추출**:
  - **Temporal Path**: GRU 레이어와 Attention 메커니즘을 사용하여 시간적 의존성을 학습한다.
  - **Multi-resolution Path**: 1D-Conv 레이어를 통해 서로 다른 해상도의 특징을 추출하고, 이를 다시 GRU와 Attention에 통과시킨다.
- **파라미터 추정**: 위 두 경로의 출력을 결합하여 Fully Connected 레이어를 통해 Student-t 분포의 파라미터인 평균 $\mu_t$, 공분산 $\Sigma_t$, 자유도 $\nu_t$, 책임도 $\alpha_t$를 예측한다.
  - $\Sigma$의 대각 성분에는 `softplus`, $\nu$에는 `scaled sigmoid`, $\alpha$에는 `softmax` 활성화 함수를 사용하여 물리적 제약 조건을 만족시킨다.
- **손실 함수**: 훈련 단계에서는 예측된 확률 밀도 함수의 음의 로그 가능도(Negative Log Likelihood, NLL)를 최소화하는 것을 목표로 한다.
$$L(x_t | x_{t-1:0}) = -\log \left( \sum_{i=1}^{c} \alpha_i f(x_t | x_{t-1:0}; \mu_i, \Sigma_i, \nu_i) \right)$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: DCASE 챌린지 Task 2에서 사용된 MIMII 및 ToyADMOS 데이터셋(Fan, Pump, Slider, Valve, Toy Conveyor, Toy Car 등 총 18개 데이터셋)을 사용하였다.
- **측정 지표**: AUC (Area Under the ROC Curve) 및 pAUC (partial AUC)를 사용하였다.
- **비교 모델**: OC-SVM, Isolation Forest, Autoencoder, GMM, MSCRED, DAGMM, DCASE Baseline 및 DCASE Winner 모델과 비교하였다.

### 2. 주요 결과

- **성능 우위**: 제안된 모델인 **RSMM-MR**은 대부분의 베이스라인 모델보다 우수한 성능을 보였으며, 특히 Slider와 Valve 데이터셋에서 AUC/pAUC가 대폭 향상되었다.
- **앙상블 효과**: RSMM-MR을 DCASE Winner 모델과 앙상블 하였을 때 모든 케이스에서 가장 뛰어난 성능을 기록하였다.
- **Ablation Study 결과**:
  - **SMM vs GMM**: Student-t 분포를 사용한 RSMM-MR이 가우시안 분포를 사용한 RGMM-MR보다 일관되게 높은 성능(특히 pAUC)을 보였다.
  - **Multi-resolution**: 다중 해상도 구조를 추가했을 때 AUC와 pAUC가 각각 평균 0.0348, 0.0757 증가하였다.
  - **Attention**: Attention 메커니즘이 없을 때보다 있을 때 시간적 역학 캡처 능력이 향상되어 성능이 더 높게 나타났다.

### 3. 강건성 테스트

훈련 데이터에 인위적으로 가우시안 노이즈 버스트(Gaussian noise bursts)를 추가하여 실험한 결과, RSMM-MR이 RGMM-MR보다 성능 하락 폭이 훨씬 적어 이상치에 대한 강건함이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 단순히 딥러닝 아키텍처를 복잡하게 만드는 것보다, **데이터의 통계적 특성(이상치 존재 가능성)을 고려한 확률 분포의 선택**이 이상 탐지 성능에 결정적인 영향을 미칠 수 있음을 보여준다.

특히 Student-t 분포의 자유도 $\nu$를 학습 가능하게 설정함으로써, 모델이 스스로 데이터의 '두꺼운 꼬리' 특성을 반영하여 정상 데이터의 경계를 더 "타이트하게" 모델링할 수 있게 된 점이 핵심이다. 이는 생성 모델이 흔히 겪는 "분포 외 데이터에 너무 높은 확률을 부여하는 문제"를 완화하는 효과적인 전략이 된다.

다만, 본 연구에서는 자유도의 범위를 1에서 10 사이로 제한하였는데, 이 하이퍼파라미터 설정이 결과에 어떤 영향을 미치는지에 대한 더 깊은 분석은 제시되지 않았다. 또한, 앙상블 모델의 성능이 매우 높게 나타나는데, 이는 단일 모델만으로는 해결하기 어려운 도메인 간 편차가 존재함을 시사한다.

## 📌 TL;DR

이 논문은 노이즈가 섞인 훈련 데이터에서도 강건하게 작동하는 오디오 이상 탐지 모델 **RSMM-MR**을 제안한다. 핵심은 **Student-t Mixture Model**을 통해 이상치에 강건한 밀도 추정을 수행하고, **다중 해상도 CNN-GRU**와 **Attention** 구조로 복잡한 시간적 패턴을 학습하는 것이다. 실험 결과, 기존의 가우시안 기반 모델이나 일반적인 오토인코더보다 뛰어난 성능을 보였으며, 특히 실제 산업 현장과 유사한 오염된 데이터 환경에서 강력한 성능을 입증하여 향후 실용적인 기계 상태 모니터링 시스템 구축에 기여할 가능성이 높다.

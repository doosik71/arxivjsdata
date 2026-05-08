# Bayesian Multivariate Nonlinear State Space Copula Models

Alexander Kreuzer, Luciana Dalla Valle, and Claudia Czado (2019)

## 🧩 Problem to Solve

본 논문은 시계열 데이터 분석에서 널리 사용되는 상태 공간 모델(State Space Model, SSM)의 한계를 극복하고자 한다. 기존의 선형 가우시안 상태 공간 모델(Linear Gaussian State Space Models)은 계산적 효율성이 높지만, 데이터가 선형성(Linearity)과 정규성(Normality)을 벗어나는 경우 적용하기 어렵다는 치명적인 제약이 있다.

특히 다변량 시계열 데이터에서는 변수 간의 복잡한 비선형 의존 구조와 꼬리 의존성(Tail Dependence)이 나타나는 경우가 많으며, 실제 환경 데이터에서는 결측치(Missing data)가 빈번하게 발생한다. 따라서 본 연구의 목표는 정규성 가정을 완화하고 비선형 및 비가우시안(Non-Gaussian) 의존 구조를 유연하게 모델링할 수 있는 다변량 상태 공간 코퓰러 모델(Multivariate State Space Copula Model)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 코퓰러(Copula) 이론과 바인 코퓰러(Vine Copula) 구조를 상태 공간 모델에 결합하는 것이다.

1. **유연한 의존성 모델링**: 관측 방정식(Observation equation)과 상태 방정식(State equation) 각각에 서로 다른 코퓰러 패밀리(Copula family)를 적용하여, 교차 단면적 의존성과 시간적 의존성을 개별적으로 정의할 수 있게 하였다.
2. **바인 코퓰러 구조의 도입**: 각 시점의 모델을 잠재 상태(Latent state)를 루트 노드로 하는 C-vine 코퓰러(첫 번째 트리에서 절단됨)로 설계하고, 잠재 상태 간의 시간적 의존성을 묘사하기 위해 D-vine 코퓰러를 사전 분포(Prior distribution)로 사용하였다.
3. **베이지안 추론 프레임워크**: 복잡한 비선형 구조로 인해 표준 칼만 필터(Kalman filter)를 사용할 수 없는 문제를 해결하기 위해, Hamiltonian Monte Carlo(HMC) 방법과 No-U-Turn sampler를 이용한 베이지안 추론 방식을 제안하였다.
4. **결측치 처리의 자연스러운 통합**: 결측치를 잠재 변수(Latent variables)로 취급함으로써, 별도의 보간법 없이 추론 과정에서 자연스럽게 처리할 수 있도록 설계하였다.

## 📎 Related Works

- **Linear Gaussian SSM**: 칼만 필터 등으로 대표되는 전통적 모델이나, 정규성 및 선형성 가정으로 인해 실제 데이터의 비정규적 특성을 반영하지 못한다.
- **Nonlinear SSM**: 일부 연구에서 비선형성을 도입했으나, 여전히 오차 항에 대해 가우시안 가정을 유지하는 경우가 많아 유연성이 부족하다.
- **Copula-based Approaches**: 코퓰러는 주변 분포(Marginals)와 의존 구조를 분리하여 모델링할 수 있게 해주며, 특히 Archimedean 코퓰러(Clayton, Gumbel 등)는 비대칭적 꼬리 의존성을 포착하는 데 유용하다.
- **Vine Copulas**: 고차원 데이터에서 bivariate 코퓰러를 기본 블록으로 사용하여 복잡한 다변량 의존성을 체계적으로 구축하는 방법이다. C-vine과 D-vine이 대표적이다.
- **기존 연구와의 차별점**: 이전 연구(Kreuzer et al., 2019)가 단변량 비선형 SSM을 제안했다면, 본 논문은 이를 다변량으로 확장하고 관측 및 상태 방정식에 서로 다른 코퓰러 패밀리를 허용함으로써 유연성을 극대화하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 (Two-Step Approach)

본 모델은 Sklar의 정리를 바탕으로 주변 분포 모델링과 의존 구조 모델링을 분리하는 2단계 접근 방식을 취한다.

- **Step 1 (Marginal Modeling)**: 각 변수에 대해 일반화 가법 모델(Generalized Additive Models, GAM)을 적용하여 공변량의 효과를 제거하고 표준화된 오차 $Z_{tj}$를 추출한다. 이때 데이터의 비정규성을 처리하기 위해 Box-Cox 변환을 사용한다.
- **Step 2 (Dependence Modeling)**: 추출된 표준화 오차들을 코퓰러 공간으로 변환($U = \Phi(Z)$)하여 제안된 다변량 코퓰러 SSM을 적용한다.

### 2. 모델 구조 및 방정식

모델은 잠재 상태 변수 $V_t$와 관측 변수 $U_{tj}$ 간의 관계를 다음과 같이 정의한다.

**관측 방정식 (Observation Equation):**
각 관측 변수 $U_{tj}$와 잠재 상태 $V_t$의 관계는 각 변수별로 선택된 코퓰러 패밀리 $m_{obs,j}$에 의해 결정된다.
$$(U_{tj}, V_t) \sim C_{m_{obs,j}}^{U_j, V}(\cdot, \cdot; \tau_{obs,j})$$

**상태 방정식 (State Equation):**
잠재 상태 $V_t$의 시간적 전이는 코퓰러 패밀리 $m_{lat}$에 의해 결정된다.
$$(V_t, V_{t-1}) \sim C_{m_{lat}}^{V_2, V_1}(\cdot, \cdot; \tau_{lat})$$

여기서 $\tau$는 Kendall's $\tau$ 파라미터이며, 코퓰러 패밀리 $M$에는 Gaussian, Student-t, Clayton, Gumbel 등이 포함된다.

### 3. 베이지안 추론 및 학습 절차

전통적인 필터링 기법을 사용할 수 없으므로, STAN 소프트웨어의 No-U-Turn sampler(HMC의 확장형)를 사용하여 사후 분포에서 샘플링한다.

- **Likelihood**: 결측치를 잠재 변수로 처리하여 관측된 데이터에 대한 우도 함수를 구성한다.
- **Prior**: 잠재 상태의 시간적 의존성을 위해 D-vine 구조의 사전 분포를 사용하며, 파라미터 $\tau$에 대해서는 Beta 또는 Uniform 분포를 사용한다.
- **Discrete Parameter Handling**: HMC는 연속 변수만 처리 가능하므로, 코퓰러 패밀리 지표($m$)에 대해서는 먼저 연속 파라미터를 통합(summation)하여 샘플링한 후, 조건부 확률을 통해 이산 파라미터를 업데이트하는 방식을 사용한다.

### 4. 예측 절차

- **In-Sample**: 결측치가 있는 시점의 값을 사후 분포에서 샘플링하여 복원한다.
- **Out-of-Sample**: $t > T$ 인 시점에 대해 상태 방정식의 코퓰러를 통해 $V_t$를 재귀적으로 샘플링하고, 이를 다시 관측 방정식에 대입하여 $U_{tj}$를 예측한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 이탈리아 도시의 대기 오염 물질(CO, NOx, NO2) 측정 데이터.
- **특징**: 고정밀 센서(Ground Truth, GT)와 저가형 센서(Low-cost, LC)의 데이터가 혼합되어 있으며, GT 데이터에 상당한 결측치가 존재한다.
- **비교 대상**:
  - Gaussian State Space Model (가우시안 SSM)
  - Bayesian Additive Regression Trees (BART, 기계학습 기반 앙상블 모델)
- **평가 지표**: Continuous Ranked Probability Score (CRPS). 값이 낮을수록 예측 분포가 실제 값에 가깝음을 의미한다.

### 2. 주요 결과

- **정성적 분석**: 데이터의 이변량 윤곽선(Contour plot) 분석 결과, 비대칭적 의존 구조가 발견되어 제안 모델의 비가우시안 코퓰러 적용의 정당성이 확인되었다.
- **정량적 분석**:
  - CO와 NO2의 경우, 제안된 **Joint Copula SSM**이 가우시안 SSM과 BART보다 낮은 CRPS를 기록하며 가장 우수한 성능을 보였다.
  - 특히 저가형 센서 데이터를 이용해 고정밀 센서 값을 복원하는 작업에서 높은 정확도를 나타냈다.
  - NOx의 경우에는 예측 성능이 상대적으로 낮았는데, 이는 해당 물질의 의존 구조가 시간에 따라 변하기 때문으로 분석되었으며, 이는 향후 연구 과제로 제시되었다.

## 🧠 Insights & Discussion

### 강점 및 기여

- **유연성**: 주변 분포(GAM) $\rightarrow$ 다변량 의존성(C-vine) $\rightarrow$ 시간적 전이(D-vine)로 이어지는 구조를 통해 데이터의 특성에 맞는 최적의 코퓰러 조합을 찾을 수 있다.
- **결측치 처리**: 결측치를 단순한 빈칸이 아닌 추론해야 할 잠재 변수로 취급함으로써 데이터 손실을 최소화하고 다른 변수와의 상관관계를 통해 값을 효과적으로 복원하였다.

### 한계 및 비판적 해석

- **파라미터 고정 가정**: 본 모델은 Kendall's $\tau$가 시간에 따라 일정하다고 가정한다. 하지만 실험 결과(NOx 사례)에서 보듯, 실제 환경 데이터의 의존 구조는 동적으로 변할 수 있다. 이를 해결하기 위해 $\tau$를 시변 파라미터로 확장할 필요가 있다.
- **계산 복잡도**: HMC 기반의 베이지안 추론은 칼만 필터와 같은 분석적 해법에 비해 계산 비용이 매우 높다. 데이터의 규모가 커질 경우 실시간 예측에 적용하기에는 한계가 있을 수 있다.
- **단일 요인 구조**: 모델이 하나의 잠재 상태($V_t$)에 의존하는 단일 요인(Single factor) 구조를 가지고 있다. 다변량 데이터의 복잡성을 모두 담기에는 요인(Factor)의 개수를 늘리는 다요인 모델로의 확장이 필요해 보인다.

## 📌 TL;DR

본 논문은 기존 선형 가우시안 상태 공간 모델의 제약을 극복하기 위해, **C-vine 및 D-vine 코퓰러 구조를 결합한 다변량 비선형 SSM**을 제안하였다. GAM을 통한 주변 분포 모델링과 HMC 기반의 베이지안 추론을 통해 비정규적 의존성과 결측치 문제를 동시에 해결하였으며, 대기 오염 데이터 예측 실험을 통해 가우시안 SSM 및 BART보다 뛰어난 예측 성능(낮은 CRPS)을 입증하였다. 이 연구는 복잡한 비선형 시계열 데이터의 상태 추정과 예측에 있어 매우 유연한 프레임워크를 제공한다.

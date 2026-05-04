# Fast Deep Mixtures of Gaussian Process Experts

Clement Etienam, Kody Law, Sara Wade, Vitaly Zankin (2020)

## 🧩 Problem to Solve

본 논문은 가우시안 프로세스(Gaussian Processes, GPs)가 가진 두 가지 주요 한계점을 해결하고자 한다. 첫째는 계산 복잡도 문제이다. 표준 GP는 공분산 행렬의 역행렬을 계산해야 하므로 데이터 수 $N$에 대해 $O(N^3)$의 비용이 발생하여 대규모 데이터셋에 적용하기 어렵다. 둘째는 모델의 유연성 문제이다. 일반적으로 GP는 정적인(stationary) 커널 함수를 사용하는데, 이는 입력 공간 전체에서 매끄러움(smoothness)이나 주기성 등의 특성이 일정하다고 가정한다. 따라서 입력 영역에 따라 함수의 성질이 변하는 비정상성(non-stationarity)이나 불연속성을 포착하는 데 한계가 있다.

본 연구의 목표는 딥러닝의 유연한 영역 분할 능력과 GP의 정교한 확률적 회귀 능력을 결합하여, 계산 효율적이면서도 높은 정확도와 신뢰할 수 있는 불확실성 정량화(Uncertainty Quantification)를 제공하는 Mixture of Experts (MoE) 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 딥러닝 기반의 Gating Network와 희소 가우시안 프로세스(Sparse GP) 전문가(Expert)를 결합한 새로운 MoE 구조를 제안한 것이다.

중심 아이디어는 입력 공간을 유연하게 분할하는 역할은 딥 신경망(DNN)이 담당하게 하고, 분할된 각 지역 내에서의 함수 근사와 불확실성 추정은 GP 전문가가 수행하게 하는 것이다. 또한, 최대 사후 확률(MAP) 추정치를 매우 빠르게 근사할 수 있는 'Cluster-Classify-Regress (CCR)'라는 원패스(one-pass) 알고리즘을 도입하여, 모델의 학습 및 추론 속도를 획기적으로 개선하였다.

## 📎 Related Works

기존의 GP 확장 방식으로는 전문가들의 예측치를 곱하는 Product of Experts (PoEs) 방식(예: BCM, gPoE, rBCM)이 제안되었다. PoEs는 가우시안 성질을 유지하여 추론이 빠르다는 장점이 있으나, 이론적 분석에 따르면 적응적 설정(adaptive setting)에서 한계가 있다. 반면, MoE는 전문가들의 예측치를 가중 평균하는 방식으로, 통계적으로 더 견고하며 입력 공간의 비정상성과 다봉성(multi-modality)을 더 잘 처리할 수 있다.

기존 MoE 모델들은 Gating Network로 선형 분류기나 트리 기반 모델을 사용했으나, 이는 영역 분할의 유연성이 떨어지는 문제가 있었다. 또한 DNN을 전문가로 사용하는 Mixture Density Networks (MDN)의 경우, 데이터가 적은 지역에서 과적합(overfitting)이 발생하기 쉽고, 불확실성을 과소평가하여 과잉 확신(overconfident)하는 경향이 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
제안된 모델은 $L$개의 Sparse GP 전문가와 하나의 DNN Gating Network로 구성된다. 입력 $x_i$가 주어졌을 때 출력 $y_i$의 조건부 밀도는 다음과 같은 혼합 모델로 정의된다.

$$y_i | x_i \sim \sum_{l=1}^L w_l(x_i; \psi) \mathcal{N}(y_i | f(x_i; \theta_l), \sigma_l^2)$$

여기서 $w_l(x_i; \psi)$는 $l$번째 전문가를 선택할 확률을 출력하는 Gating Network이며, $f(x_i; \theta_l)$은 해당 전문가의 회귀 함수이다.

### 2. DNN Gating Network
Gating Network는 Feedforward DNN으로 구현되며, 마지막 층에 Softmax 함수를 사용하여 각 전문가의 가중치 합이 1이 되도록 한다. 

$$w_l(x; \psi) = \frac{\exp(h_l(x; \psi))}{\sum_{j=1}^L \exp(h_j(x; \psi))}$$

DNN은 ReLU 활성화 함수를 사용하며, 이를 통해 입력 공간을 매우 유연하고 복잡한 형태로 분할할 수 있다.

### 3. Sparse GP Experts
계산 비용을 줄이기 위해 각 전문가는 $M_l < N_l$개의 유도 지점(inducing points) $\tilde{x}_l$과 유도 타겟(pseudo-targets) $\tilde{f}_l$을 사용하는 Sparse GP를 채택한다. 구체적으로는 FITC (Fully Independent Training Conditional) 근사법을 사용하여 각 데이터 포인트의 예측값 $\hat{\mu}_{l,i}$와 분산 $\lambda_{l,i}$를 다음과 같이 계산한다.

$$\hat{\mu}_{l,i} = \mu_l + (k_{l,M_l,i})^T (K_{l,M_l})^{-1} (\tilde{f}_l - \mu_l)$$
$$\lambda_{l,i} = K_{\phi_l}(x_i, x_i) - (k_{l,M_l,i})^T (K_{l,M_l})^{-1} k_{l,M_l,i}$$

이 구조를 통해 전문가 한 명당 계산 복잡도를 $O(N_l^3)$에서 $O(N_l M_l^2)$로 낮출 수 있다.

### 4. 학습 절차 및 알고리즘
본 논문은 MAP 추정을 위해 Maximization-Maximization (MM) 알고리즘을 사용한다. 이는 다음 두 단계를 반복하는 좌표 상승법(coordinate ascent)의 일종이다.

1.  **할당 단계(Clustering):** 현재 모델 파라미터가 주어졌을 때, 각 데이터 포인트 $i$를 가장 확률이 높은 전문가 $z_i$에 할당한다.
2.  **최적화 단계(Parameter Update):** 할당된 데이터를 바탕으로 DNN Gating Network의 가중치 $\psi$와 각 GP 전문가의 하이퍼파라미터 $\theta_l, \sigma_l^2$ 및 유도 지점 $\tilde{x}_l$을 최적화한다.

특히, 이 반복 과정을 빠르게 근사하기 위해 **CCR (Cluster-Classify-Regress)** 알고리즘을 제안한다. CCR은 (1) 데이터를 먼저 클러스터링하고, (2) 이를 바탕으로 분류기를 학습시킨 뒤, (3) 각 클러스터 내에서 회귀 모델을 학습시키는 원패스 방식이다. 논문은 CCR이 MM 알고리즘의 매우 훌륭한 초기값 역할을 하며, 많은 경우 추가 반복 없이도 충분한 성능을 낸다는 것을 보여준다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** Motorcycle, NASA, Higdon, Bernholdt, kin40k, 그리고 대규모/고차원 데이터인 $\chi_{150k}$ (입력 차원 10, 데이터 수 15만 개)를 사용하였다.
- **비교 대상:** MDN, gPoE, RBCM, FastGP, BART, ORTHNAT, PPGPR, DSPP, Deep GP 등 최신 GP 및 신경망 기반 모델들과 비교하였다.
- **평가 지표:** 결정 계수 ($R^2$), 실행 시간(Wall-clock time), 경험적 커버리지(Empirical Coverage, $EC_{95}$), 95% 신뢰 구간의 평균 길이 ($\overline{CI}_{95}$)를 측정하였다.

### 2. 주요 결과
- **정확도 및 속도:** 대부분의 데이터셋에서 제안 모델(CCR 및 MM 기반)이 가장 높은 $R^2$를 기록하였다. 특히 $\chi_{150k}$ 데이터셋에서 다른 경쟁 모델들보다 압도적으로 빠른 실행 속도와 높은 정확도를 동시에 달성하였다.
- **불확실성 정량화:** 제안 모델은 $EC_{95} \ge 95\%$를 유지하면서도 신뢰 구간의 길이를 매우 짧게 유지하였다. 이는 MDN처럼 과신(overconfident)하거나 PoE 모델들처럼 지나치게 보수적인(conservative) 구간을 생성하지 않고, 정교하게 캘리브레이션된 불확실성을 제공함을 의미한다.
- **CCR의 효율성:** CCR 알고리즘은 MM2r(2회 반복 MM)보다 2~3배 빠르면서도 유사하거나 더 나은 정확도를 보였다.

## 🧠 Insights & Discussion

본 논문은 DNN의 강력한 표현력과 GP의 확률적 엄밀함을 결합함으로써, 계산 효율성과 예측 성능이라는 두 마리 토끼를 잡았다. 특히 Gating Network로 DNN을 사용한 덕분에, 기존의 선형/이차 분류기로는 불가능했던 복잡한 형태의 영역 분할이 가능해졌으며, 이는 곧 비정상성 데이터에 대한 높은 적응력으로 이어졌다.

또한, CCR 알고리즘과 MoE의 MM 알고리즘 사이의 이론적 연결 고리를 찾아낸 점이 인상적이다. 이는 단순한 휴리스틱으로 보였던 CCR 방식이 실제로는 MAP 추정의 효율적인 근사치임을 입증한 것이며, 향후 다른 MoE 구조의 빠른 학습을 위한 프레임워크로 확장될 가능성을 제시한다.

다만, 전문가의 수 $L$을 BIC(Bayesian Information Criterion) 등을 통해 사전에 결정해야 한다는 점은 한계로 남는다. 데이터에 따라 전문가 수가 자동으로 조절되는 무한 혼합 전문가(Infinite Mixture of Experts) 모델로의 확장이 향후 과제로 제시되었다.

## 📌 TL;DR

본 논문은 **DNN Gating Network**와 **Sparse GP Experts**를 결합한 고속 MoE 모델을 제안한다. DNN은 입력 공간을 유연하게 분할하고, Sparse GP는 각 지역에서 정확한 회귀와 불확실성 추정을 수행한다. 특히 **CCR 알고리즘**을 통해 학습 속도를 획기적으로 높였으며, 실험 결과 대규모 고차원 데이터셋에서도 기존 GP 및 딥러닝 모델 대비 우수한 정확도와 신뢰할 수 있는 불확실성 정량화 성능을 입증하였다. 이 연구는 실시간성이 중요하면서도 신뢰도가 필요한 복잡한 물리 시스템 모델링 등에 광범위하게 적용될 가능성이 크다.
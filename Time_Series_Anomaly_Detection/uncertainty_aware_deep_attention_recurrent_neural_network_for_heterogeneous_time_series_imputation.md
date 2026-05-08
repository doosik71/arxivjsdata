# Uncertainty-Aware Deep Attention Recurrent Neural Network for Heterogeneous Time Series Imputation

Linglong Qian, Zina Ibrahim, Richard JB Dobson (2024)

## 🧩 Problem to Solve

다변량 시계열(Multivariate Time Series) 데이터에서 결측치(Missingness)는 매우 흔하게 발생하며, 이는 후속 분석의 신뢰성을 떨어뜨리는 주요한 장애물이 된다. 특히 의료 기록(EHR)과 같은 데이터는 임상적 결정에 따라 불규칙하게 샘플링되므로 비무작위 결측(Non-random missingness) 문제가 심각하다.

기존의 최신 모델인 BRITS는 양방향 RNN(Bidirectional RNN)을 사용하여 우수한 성능을 보였으나, 다음과 같은 한계가 존재한다. 첫째, 단일 레이어 구조로 설계되어 복잡한 데이터에서 발생할 수 있는 문제를 해결하기 위한 딥 아키텍처(Deep Architecture)로의 확장이 어렵다. 둘째, 결정론적(Deterministic) 모델이기 때문에 추정된 결측값에 대해 모델이 얼마나 확신하는지를 나타내는 불확실성(Uncertainty)을 측정할 수 없다.

따라서 본 논문의 목표는 이질적인 다변량 시계열 데이터에서 결측값과 그에 따른 불확실성을 동시에 추정할 수 있는 확장 가능한 딥러닝 모델인 DEARI(DEep Attention Recurrent Imputation)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 BRITS 구조를 기반으로 하되, 이를 깊게 쌓을 수 있도록 최적화하고 자기지도 학습 및 베이즈 추론을 결합하는 것이다.

1. **확장 가능한 딥 아키텍처**: Self-Attention 메커니즘과 잔차 연결(Residual Component)을 도입하여, 기존 BRITS의 한계를 넘어 10개 레이어까지 안정적으로 쌓을 수 있는 깊은 순환 신경망 구조를 설계하였다.
2. **자기지도 메트릭 학습(Self-Supervised Metric Learning)**: 데이터의 이질성 문제를 해결하기 위해, 동일 샘플의 순방향 및 역방향 표현(Representation) 간의 유사성을 최적화하는 Deep Metric Learning(DML) 프레임워크를 적용하여 데이터 표현 능력을 높였다.
3. **불확실성 정량화**: Bayesian Marginalization 전략을 통해 모델을 베이즈 신경망(BNN)으로 변환함으로써, 추정값과 함께 신뢰할 수 있는 신뢰 구간(Confidence Bounds)을 제공하는 Stochastic DEARI를 구현하였다.

## 📎 Related Works

기존의 다변량 시계열 결측치 보간 연구는 크게 다음과 같이 분류된다.

- **RNN 기반 모델**: GRUD는 시간적 쇠퇴(Temporal Decay) 개념을 도입하여 과거 관측값이 현재 결측치에 미치는 영향이 시간에 따라 감소함을 모델링하였다. MRNN과 BRITS는 이를 확장하여 특징 간의 상관관계와 양방향 시간적 역학을 모두 포착하였다. 특히 BRITS는 데이터 특성에 대한 가정을 최소화하여 SOTA 성능을 달성하였다.
- **확률론적 모델**: V-RIN(VAE 기반), $E^2GAN$(GAN 기반) 등이 불확실성을 다루기 위해 제안되었다. 그러나 이러한 모델들은 별도의 불확실성 모듈을 추가해야 하므로 학습이 불안정하거나 결합도가 높아져 오히려 보간 정확도가 떨어지는 경향이 있다.

본 논문은 이러한 기존 연구와 달리, 보간 작업을 다운스트림 작업(분류/회귀)과 동시에 수행하지 않고 독립적인 전처리 모듈로 분리한다. 이는 거친 입도의 분류 작업과 세밀한 보간 작업이 서로 다른 데이터 특성에 집중함으로써 발생할 수 있는 모델 편향(Model Bias)을 방지하기 위함이다.

## 🛠️ Methodology

### 1. BRITS Backbone

DEARI는 BRITS의 기본 가정을 따른다. 핵심은 시간적 쇠퇴(Temporal Decay)와 특징 간 상관관계를 동시에 이용하는 것이다.

- **시간적 쇠퇴**: 시간 간격 $\delta_t$가 클수록 이전 상태의 영향력이 감소하며, 쇠퇴 인자 $\gamma_t$는 다음과 같이 계산된다.
    $$\gamma_t = \exp\{-\max(0, W_{\gamma} \delta_t + b_{\gamma})\}$$
- **결측치 보완**: 쇠퇴된 은닉 상태 $\hat{h}_{t-1}$를 통해 역사적 표현 $\hat{x}_t$를 생성하고, 이를 마스크 $m_t$와 결합하여 보완 벡터 $x_{hc}^t$를 만든다.
    $$x_{hc}^t = m_t \odot x_t + (1 - m_t) \odot \hat{x}_t$$
- **특징 간 추정**: 완전 연결 층을 통해 현재 시점의 다른 특징들로부터 결측치를 추정하는 $x_{fc}^t$를 계산하며, 최종 보간값 $C_t$는 시간적 추정치와 특징 간 추정치를 가중 결합하여 생성된다.

### 2. Deep Attention Recurrent Neural Network

단순히 RNN 레이어를 쌓는 것은 결측치로 인한 오차를 증폭시킬 수 있다. 이를 해결하기 위해 DEARI는 다음을 도입한다.

- **Self-Attention 초기화**: 각 레이어 $l$의 초기 은닉 상태 $h_0^l$를 결정하기 위해, 이전 레이어 $l-1$의 모든 시점 은닉 상태들을 $[CLS]$ 토큰과 함께 결합하여 Self-Attention(MSA)과 Feed-Forward Network(FFN)를 통과시킨다.
    $$\hat{h}_{l-1} = [CLS, h_0^{l-1}, h_1^{l-1}, \dots, h_t^{l-1}]$$
    이 과정을 통해 저수준의 세밀한 정보와 고수준의 전역적 정보를 모두 통합한 임베딩을 생성한다.
- **Residual Component**: 깊은 층에서도 학습의 안정성을 확보하기 위해, 각 레이어가 원본 데이터 $X$에 직접 접근할 수 있는 잔차 연결 구조를 가진다.

### 3. Deep Self-Supervised Metric Learning (S$^2$DML)

데이터의 이질성을 극복하기 위해, 동일 샘플의 순방향 표현($R_A$)과 역방향 표현($R_P$)을 긍정 쌍(Positive pair)으로, 다른 샘플의 표현($R_N$)을 부정 쌍(Negative pair)으로 설정하는 트리플렛(Triplet) 구조를 사용한다. 이때 Multi-Similarity(MS) 손실 함수를 사용하여 정보량이 많은 샘플에 더 큰 가중치를 두어 최적화한다.
$$L = \frac{1}{|B|} \sum_{i \in B} \left( \frac{1}{\alpha} \log[1 + \sum_{n \in N_i} e^{\alpha(S_{in}-\epsilon)}] + \frac{1}{\beta} \log[1 + \sum_{p \in P_i} e^{\beta(S_{ip}-\epsilon)}] \right)$$

### 4. Bayesian Marginalization Strategy

모델의 파라미터를 단일 값이 아닌 가우시안 분포에서 샘플링하도록 하여 인식적 불확실성(Epistemic Uncertainty)을 포착한다.

- **파라미터 분포**: 가중치 $W$를 $\mu_W$와 $\rho_W$로 정의된 분포에서 샘플링하여 사용한다.
- **학습 전략**: BNN의 학습 불안정성을 해결하기 위해 'Freeze(최적화)'와 'Unfreeze(불확실성 탐색)' 상태를 교차하는 전략을 사용하여 탐색(Exploration)과 이용(Exploitation)의 균형을 맞춘다.

## 📊 Results

### 실험 설정

- **데이터셋**: Beijing Air Quality, eICU, MIMIC-III (두 종류), PhysioNet Challenge 2012, Traffic 등 총 5개의 실제 데이터셋을 사용하였다.
- **비교 대상**: BRITS, GRUD, V-RIN, MRNN.
- **평가 지표**: 평균 절대 오차(MAE, Mean Absolute Error) 및 평균 상대 오차(MRE, Mean Relative Error).
- **설정**: 5-fold 교차 검증을 수행하였으며, 결측률을 5%, 10%, 20%로 설정하여 테스트하였다.

### 주요 결과

- **보간 성능**: DEARI는 모든 데이터셋에서 BRITS 및 다른 베이스라인 모델보다 우수한 성능(낮은 MAE/MRE)을 보였다.
- **모델 변형**: Deterministic DEARI와 Stochastic(Bayesian) DEARI 모두 SOTA 모델들을 능가하였으며, 특히 대규모 데이터셋에서는 Bayesian DEARI의 성능이 더 뛰어났다.
- **Ablation Study**:
  - **깊이(Depth)**: 단순 적층(Multi-layer BRITS)은 성능 향상이 미비했으나, DEARI는 레이어 수가 증가함에 따라 성능이 뚜렷하게 향상되었으며, 10개 레이어에서 최적의 성능을 보였다.
  - **DML 효과**: DML은 보간 정확도를 더욱 높였으며, 특히 Self-Attention과 결합했을 때 효과가 극대화되었다.
  - **신뢰 구간**: Bayesian 모델이 생성한 신뢰 구간은 관측값에 가까울수록 좁아지는 특성을 보였는데, 이는 시간적 쇠퇴 원칙과 일치함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 다변량 시계열 보간에서 **'확장성'**과 **'불확실성'**이라는 두 가지 핵심 난제를 성공적으로 해결하였다. 특히, 단순히 층을 깊게 쌓는 것이 아니라 Attention 메커니즘을 통해 레이어 간 정보를 효율적으로 전달함으로써 딥 아키텍처의 이점을 취한 점이 인상적이다. 또한, 보간을 다운스트림 태스크와 분리하여 전처리 모듈로 정의함으로써 태스크 특화 편향을 제거한 접근 방식은 타당해 보인다.

다만, 모델의 복잡도(파라미터 수)가 레이어 수에 따라 선형적으로 증가하며, 성능 향상 폭은 점차 둔화되는 한계가 있다. 저자들 또한 BRITS 백본 자체가 모델의 일반화 능력을 제한하는 요소가 될 수 있음을 언급하며, 향후 순수 Attention 기반 모델로의 확장을 제시하고 있다.

## 📌 TL;DR

본 논문은 BRITS를 기반으로 **Self-Attention, Deep Metric Learning, Bayesian Marginalization**을 결합한 **DEARI** 모델을 제안한다. 이를 통해 (1) 깊은 신경망 구조를 통한 복잡한 시계열 데이터의 정밀한 보간, (2) 자기지도 학습을 통한 데이터 표현력 향상, (3) 베이즈 추론을 통한 추정값의 불확실성 정량화를 달성하였다. 결과적으로 다양한 실제 데이터셋에서 기존 SOTA 모델들을 능가하는 성능을 입증하였으며, 신뢰할 수 있는 결측치 보간 프레임워크를 제공한다.

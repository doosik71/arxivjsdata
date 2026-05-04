# Model-based Offline Imitation Learning with Non-expert Data

Jeongwon Park, Lin Yang (2022)

## 🧩 Problem to Solve

본 논문은 Imitation Learning(IL)에서 발생하는 두 가지 주요 문제인 **compounding errors(복합 오류)**와 **전문가 데이터 획득의 고비용 문제**를 해결하고자 한다.

전통적인 Behavioral Cloning(BC)은 단순하고 확장성이 좋지만, 학습 시 겪지 못한 상태에 도달했을 때 발생하는 covariate shift로 인해 시간이 지남에 따라 오류가 누적되는 compounding errors 문제가 발생한다. 이를 해결하기 위한 Adversarial IL 방식들은 성능이 뛰어나지만, 환경과의 직접적인 상호작용(interaction)이 필수적이며, 이는 실제 환경(예: 의료, 자율주행)에서 매우 위험하거나 비용이 많이 드는 작업일 수 있다.

또한, 기존의 많은 IL 방법론들은 오직 최적(optimal)의 전문가 데이터셋만을 사용한다. 그러나 실제 환경에서는 최적의 데이터보다 하위 최적(suboptimal)의 데이터가 훨씬 더 많고 획득하기 쉽다. 따라서 본 논문의 목표는 **환경과의 상호작용 없이(offline), 전문가 데이터와 하위 최적 데이터를 모두 활용하여 전문가의 성능을 효과적으로 모방하는 모델 기반 offline imitation learning 프레임워크를 제안**하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **하위 최적 데이터를 포함한 전체 데이터셋으로 환경의 역학(dynamics)을 학습한 모델($T^l$)을 구축하고, 이 가상 모델 내부에서 정책을 적대적으로(adversarially) 학습시키는 것**이다.

중심적인 직관은 다음과 같다.

1. 전문가 데이터만으로는 상태-행동 공간의 커버리지가 좁아 신뢰할 수 있는 모델을 만들기 어렵지만, 하위 최적 데이터를 함께 사용하면 더 넓은 영역의 dynamics 모델을 학습할 수 있다.
2. 이렇게 학습된 모델 내에서 정책을 훈련시키면, 실제 환경과의 상호작용 없이도 전문가의 상태 분포($d^{\pi^*}_T$)와 학습자의 상태 분포($d^{\hat{\pi}}_T$) 사이의 차이를 직접 줄임으로써 BC의 고질적인 문제인 covariate shift를 완화할 수 있다.
3. 모델의 부정확성으로 인한 정책의 과적합(exploitation) 문제는 uncertainty penalization을 통해 해결한다.

## 📎 Related Works

- **Behavioral Cloning (BC):** 전문가의 행동을 그대로 복제하는 방식으로 단순하지만, 시간 지평(time horizon)에 대해 오차가 이차 함수적으로 증가하는 특성이 있다.
- **Adversarial Imitation Learning (AIL):** GAIL 등은 전문가와 학습자의 분포 차이를 줄이는 min-max 최적화를 사용한다. 성능은 좋으나 환경과의 상호작용이 필수적이라는 한계가 있다.
- **Offline Reinforcement Learning (Offline RL):** 고정된 데이터셋만 사용하지만, 보상 함수(reward function)가 존재해야 한다는 전제가 필요하다. 본 논문은 보상 함수 대신 전문가 데이터를 사용한다는 점에서 차별화된다.
- **Learning from Suboptimal Demonstrations:** 하위 최적 데이터에서 학습하려는 시도가 있었으나, 완전한 offline 설정에서 이를 체계적으로 활용한 연구는 제한적이었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 논문은 먼저 행동 데이터셋(전문가 데이터 $M$개 + 하위 최적 데이터 $N$개)을 사용하여 dynamics 모델 $T^l$을 학습한다. 이후 이 모델을 이용해 가상 롤아웃(rollout)을 생성하고, 판별자(discriminator)와 정책(policy)을 적대적으로 학습시킨다.

### 2. Algorithm 2: 제안하는 학습 방법

논문은 이론적 분석을 통해 도출된 Algorithm 1을 개선하여, BC 손실 함수와 판별자 손실 함수를 결합한 **Algorithm 2**를 제안한다.

- **목표 함수:** 정책 $\hat{\pi}$는 다음의 손실 함수를 최소화한다.
$$\text{loss}(\pi) = \frac{1}{M} \sum_{m=1}^{M} ||\pi^*(a|s^*) - \pi(a|s^*)||_1 - \mathbb{E}_{s \sim d^{\pi}_{T^l}} [\hat{f}(s)]$$
여기서 $\hat{f}$는 전문가의 상태 분포와 학습자의 상태 분포를 구분하는 판별자이다.

- **판별자 $\hat{f}$의 학습:** 판별자는 다음과 같이 최적화된다.
$$\hat{f} = \text{argmax}_{f \in F} \left( \frac{1}{M} \sum_{m=1}^{M} f(s^*) - \mathbb{E}_{s \sim d^{\pi}_{T^l}} [f(s)] \right)$$

- **작동 원리:**
    1. 첫 번째 항은 전문가의 행동을 직접 따라하게 하는 BC loss이다.
    2. 두 번째 항은 학습자가 생성하는 상태 분포가 전문가의 상태 분포와 유사해지도록 유도한다. 이는 $\text{TV}$ distance를 직접 줄이는 효과를 가져와 covariate shift를 억제한다.

### 3. Model Uncertainty 및 Penalization

학습된 모델 $T^l$이 부정확한 영역에서 정책이 비정상적인 이득을 취하는 **model exploitation** 문제를 해결하기 위해, 모델의 불확실성을 측정하고 이를 페널티로 부여한다.

- **불확실성 추정:** $K$개의 확률적 신경망(probabilistic neural networks) 앙상블을 학습시켜 각 예측값의 분산(covariance matrix $\Sigma$)을 통해 불확실성 $\hat{U}(s, a)$를 계산한다.
- **페널티 적용:** 판별자 함수에 불확실성을 뺀 $\hat{f}_u(s, a) = f(s, a) - \lambda \hat{U}(s, a)$를 사용하여, 모델이 불확실한 영역으로 정책을 유도하는 것을 방지한다.

### 4. 학습 절차

1. **모델 학습:** 전문가 및 하위 최적 데이터를 사용하여 앙상블 dynamics 모델 $T^l$을 사전 학습한다.
2. **가상 롤아웃:** 현재 정책 $\hat{\pi}$를 $T^l$에서 실행하여 상태-행동 데이터를 수집한다.
3. **판별자 업데이트:** 수집된 데이터와 전문가 데이터를 사용하여 $\hat{f}$를 업데이트한다.
4. **정책 업데이트:** PPO(Proximal Policy Optimization)를 사용하여 BC loss와 판별자 보상을 최적화한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** OpenAI Gym의 MuJoCo 시뮬레이터(Hopper, HalfCheetah, Walker2d)를 사용한다.
- **비교 대상:** Behavioral Cloning (BC), Online Adversarial IL.
- **변수:** 전문가 궤적의 수($M = 1, 3, 10, 30$)를 조절하며 성능을 측정한다.
- **데이터 구성:**
  - `Algorithm2_w`: 학습 능력이 어느 정도 있는(medium performance) 정책으로 수집한 하위 최적 데이터 사용.
  - `Algorithm2_n`: 완전한 랜덤 정책으로 수집한 데이터 사용.

### 2. 정량적 결과

- **저데이터 환경에서의 우위:** 전문가 데이터가 매우 적은 상황($M=1, 3$)에서 Algorithm 2는 BC보다 압도적으로 높은 성능을 보인다.
- **데이터 커버리지의 중요성:** `Algorithm2_w`가 `Algorithm2_n`보다 성능이 좋으며, 이는 하위 최적 데이터의 질(coverage)이 모델의 정확도와 정책 성능에 직접적인 영향을 미침을 시사한다.
- **Online AIL과의 비교:** 놀랍게도 `Algorithm2_w`는 실제 환경과 상호작용하며 학습한 Adversarial IL과 거의 유사한 성능을 달성하였다.
- **데이터 효율성:** BC는 $M$이 증가함에 따라 성능이 가파르게 상승하지만, Algorithm 2는 적은 $M$에서도 이미 높은 성능을 보여 전문가 데이터 의존도가 훨씬 낮음을 입증하였다.

### 3. 정성적 결과 (Recovery Behavior)

- 불확실성 측정 지표 $\tilde{U}_t$를 통해 분석한 결과, BC는 시간이 지남에 따라 전문가 분포에서 빠르게 멀어지지만, Algorithm 2는 초기 상태가 전문가 분포 밖(하위 최적 데이터 영역)일지라도 다시 전문가의 상태 분포로 돌아오는 **recovery behavior**를 보였다.

## 🧠 Insights & Discussion

### 강점

- **이론적 근거:** 단순한 실험적 제안이 아니라, 전문가 샘플 수 $M$에 대한 suboptimality의 의존성을 선형적으로 낮출 수 있음을 수학적으로 증명하였다.
- **실용성:** 보상 함수를 설계하기 어려운 환경에서, 하위 최적 데이터만으로도 전문가 수준의 성능을 낼 수 있는 offline 프레임워크를 제시하였다.

### 한계 및 가정

- **데이터 커버리지 가정:** 행동 데이터셋이 상태-행동 공간을 충분히 커버하고 있다는 가정($C$가 유한함)에 의존한다. 커버리지가 매우 낮은 경우(예: 완전 랜덤 데이터) 성능이 저하된다.
- **불확실성 추정의 휴리스틱:** 앙상블을 이용한 불확실성 추정 방식은 실용적이지만 이론적으로 완벽하게 정교한 방법은 아니며, 하이퍼파라미터 $\lambda$에 영향을 받는다.

### 비판적 해석

본 연구는 Offline RL의 pessimism(비관론) 개념을 Offline IL에 성공적으로 이식하였다. 특히, "전문가 데이터를 더 많이 모으는 것"보다 "하위 최적 데이터로 환경 모델을 잘 만드는 것"이 저데이터 환경에서 더 효율적일 수 있음을 보여준 점이 매우 인상적이다. 다만, 실제 복잡한 고차원 이미지 기반 환경에서도 이러한 dynamics 모델링과 uncertainty penalization이 유효하게 작동할지는 추가적인 검증이 필요해 보인다.

## 📌 TL;DR

본 논문은 **하위 최적 데이터를 활용해 학습된 가상 환경(dynamics model) 내에서 정책을 적대적으로 학습시키는 Offline Imitation Learning 프레임워크**를 제안한다. 이를 통해 전문가 데이터가 매우 적은 상황에서도 BC의 고질적인 문제인 compounding errors를 해결하고, Online Adversarial IL에 근접하는 성능을 달성하였다. 이 연구는 보상 함수 없이도 하위 최적 데이터의 가치를 극대화하여 실제 적용 가능성을 높인 중요한 연구이다.

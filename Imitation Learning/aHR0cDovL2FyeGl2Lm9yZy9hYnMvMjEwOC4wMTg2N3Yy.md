# A Pragmatic Look at Deep Imitation Learning

Kai Arulkumaran, Dan Ogawa Lillrank (2023)

## 🧩 Problem to Solve

본 논문은 심층 강화학습(Deep Reinforcement Learning, DRL) 분야에서 지속적으로 제기되어 온 재현성 위기(reproducibility crisis) 문제를 모방 학습(Imitation Learning, IL) 관점에서 다룬다. 기존의 많은 IL 알고리즘들은 서로 다른 데이터셋, 베이스 RL 알고리즘, 평가 설정 및 하이퍼파라미터 최적화 수준을 가지고 있어, 어떤 알고리즘이 실제로 더 우수한 성능을 보이는지 공정하게 비교하기가 매우 어려운 상황이다.

특히, 초기 GAIL과 같은 알고리즘들은 On-policy RL을 기반으로 설계되었으나, 최근에는 데이터 효율성이 높은 Off-policy RL 알고리즘으로의 전환이 이루어지고 있다. 따라서 본 논문의 목표는 6가지 주요 IL 알고리즘을 동일한 Off-policy 베이스 알고리즘인 Soft Actor-Critic (SAC) 기반으로 재구현하고, 동일한 하이퍼파라미터 최적화 예산을 할당하여 MuJoCo 벤치마크 환경에서 공정하게 성능을 비교 분석하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 이론적인 새로운 알고리즘 제안보다는, 실무적(pragmatic) 관점에서 기존 알고리즘들을 동일한 조건 아래 재평가하여 실질적인 가이드라인을 제공하는 데 있다.

- **통일된 벤치마크 프레임워크 구축**: 서로 다른 IL 알고리즘들을 동일한 베이스 RL 알고리즘(SAC)과 데이터셋(D4RL) 상에서 비교할 수 있는 오픈 소스 코드베이스를 구축하였다.
- **Off-policy 기반의 재구현**: On-policy 기반이었던 GMMIL, RED, DRIL 알고리즘을 Off-policy 방식으로 업데이트하여 현대적인 RL 설정에서의 성능을 검증하였다.
- **공정한 하이퍼파라미터 최적화**: 모든 알고리즘에 동일한 베이지안 최적화(Bayesian optimization) 예산을 부여하여, 특정 알고리즘만 과도하게 튜닝되는 편향을 제거하였다.
- **데이터 양에 따른 성능 분석**: 전문가 궤적(expert trajectories)의 수(5, 10, 25개)를 달리하여, 데이터 가용성에 따른 알고리즘별 효율성을 분석하였다.

## 📎 Related Works

논문에서는 모방 학습의 접근 방식을 크게 세 가지로 분류하여 설명한다.

1. **Behavioral Cloning (BC)**: 전문가의 상태-행동 쌍을 지도 학습(Supervised Learning) 방식으로 학습하는 가장 단순한 방법이다. 하지만 전문가가 방문하지 않은 상태에 진입했을 때 발생하는 복합 오류(compounding errors) 문제로 인해 성능이 급격히 저하되는 한계가 있다.
2. **Inverse Reinforcement Learning (IRL)**: 전문가의 행동을 통해 보상 함수 $\hat{R}$을 추론하고, 이를 최대화하는 정책 $\hat{\pi}$를 학습하는 방식이다.
3. **Adversarial Imitation Learning**: GAIL과 같이 판별자(Discriminator)를 통해 전문가와 학습자의 분포 차이를 학습하여 보상으로 사용하는 방식이다.
4. **Distribution Matching IL**: 적대적 학습의 불안정성을 피하기 위해 MMD(Maximum Mean Discrepancy), Wasserstein 거리, Moment matching 등 수학적 분포 거리 측정 방식을 사용하여 전문가의 상태-행동 분포를 모방하는 방식이다.

기존 연구들은 대부분 자신들이 제안한 알고리즘의 우수성을 입증하기 위해 최적화된 설정을 사용하거나, 전문가 데이터를 서브샘플링하여 BC의 성능을 의도적으로 낮추는 등 공정한 비교가 이루어지지 않았다는 점을 지적한다.

## 🛠️ Methodology

### 전체 시스템 구조

모든 IL 알고리즘은 베이스 RL 알고리즘으로 **Soft Actor-Critic (SAC)**을 사용한다. SAC는 Maximum Entropy RL 프레임워크를 사용하여 탐색 효율성을 높이며, Off-policy 방식으로 경험 재현 버퍼(experience replay buffer)를 활용한다.

### 주요 구성 요소 및 알고리즘별 보상 함수 $\hat{R}$

각 알고리즘은 전문가 데이터 $\xi^*$와 학습자의 궤적 $\hat{\pi}$ 사이의 간극을 줄이기 위해 서로 다른 보상 함수를 정의한다.

- **BC**: RL 없이 직접 최적화를 수행하며, 전문가 행동 $a^*$와의 차이를 최소화한다.
  $$\text{argmin}_{\theta} \mathbb{E}_{s, a^* \sim \xi^*} [L(a^*, \hat{\pi}(a|s; \theta))]$$
- **GAIL**: 판별자 $D(s, a)$를 학습시켜 전문가와 학습자를 구분하며, 보상은 $\log D(s, a)$ 또는 $-\log(1 - D(s, a))$ 형태로 주어진다.
- **AdRIL**: 전문가-학습자 간의 Moment matching을 수행하며, 다음과 같은 보상 함수를 사용한다.
  $$\hat{R} = \begin{cases} 1/|\xi^*| & \text{if } (s, a) \in \xi^* \\ 0 & \text{if } (s, a) \sim \hat{\pi} \\ -1/(\text{round} \cdot |\xi|) & \text{if } (s, a) \sim \hat{\pi}_{\text{old}} \end{cases}$$
- **GMMIL**: MMD(Maximum Mean Discrepancy)를 이용하여 커널 기반의 분포 거리를 측정하고 이를 보상으로 사용한다.
- **RED**: Random Network Distillation(RND)을 활용하여 전문가 데이터의 서포트(support)를 추정하고, MSE 기반의 보상을 생성한다.
- **DRIL**: BC 정책들의 앙상블에서 발생하는 분산(variance)을 통해 불확실성을 측정하고, 이를 기반으로 보상을 부여한다.
- **PWIL**: Wasserstein-2 거리를 최소화하며, Greedy coupling을 통해 온라인으로 계산된 전송 비용(transport cost)을 보상으로 변환한다.

### 학습 및 추론 절차

1. **데이터 준비**: D4RL의 'expert-v2' 데이터셋을 사용하며, 터미널 상태를 정확히 처리하기 위해 **흡수 상태 지표(absorbing state indicator)**를 상태 벡터에 추가한다.
2. **하이퍼파라미터 최적화**: 각 궤적 예산(5, 10, 25)마다 베이지안 최적화를 통해 30회의 시도를 수행하여 최적의 하이퍼파라미터를 찾는다.
3. **평가**: 최적의 하이퍼파라미터를 적용하여 10개의 시드(seed)로 학습시킨 후, IQM(Interquartile Mean) 지표로 성능을 측정한다.

## 📊 Results

### 실험 설정

- **환경**: MuJoCo (Ant, HalfCheetah, Hopper, Walker2D)
- **데이터 예산**: 전문가 궤적 수 5, 10, 25개
- **평가 지표**: 정규화된 점수 (Normalised Score) 및 IQM $\pm$ 95% 신뢰구간

### 주요 결과

- **GAIL의 강건함**: GAIL은 모든 데이터 예산 범위에서 일관되게 높은 성능을 보였다. 이는 그동안 많은 후속 연구들이 GAIL의 안정성과 성능을 개선하는 데 집중했기 때문으로 분석된다.
- **AdRIL의 효율성**: 데이터 예산이 많을 때(25개) AdRIL은 GAIL과 대등하거나 더 나은 성능을 보였으며, 구현 및 튜닝이 훨씬 간단하다는 장점이 있다.
- **BC의 재발견**: 데이터가 충분할수록 BC는 매우 강력한 베이스라인이 된다. 특히 25개 궤적에서는 많은 RL 기반 IL 알고리즘들과 경쟁 가능한 수준에 도달한다.
- **PWIL의 성능**: PWIL 또한 전반적으로 준수한 성능을 보였으나, 데이터가 많아질수록 보상 계산 비용이 증가하는 계산적 오버헤드가 발생한다.
- **Off-policy 전환의 실패**: DRIL, GMMIL, RED의 경우 Off-policy 버전으로 전환했을 때 하이퍼파라미터 최적화에 실패하였다. 저자들은 On-policy에서 Off-policy로의 전환 과정에서 근본적인 문제가 발생한 것으로 추측한다.

## 🧠 Insights & Discussion

본 논문은 알고리즘의 이론적 우수성보다 **구현 디테일과 하이퍼파라미터 튜닝이 실제 성능에 더 큰 영향을 미칠 수 있음**을 시사한다.

- **실무적 추천**:
  - 가장 먼저 **BC**를 시도하여 베이스라인을 잡을 것을 권장한다.
  - 데이터가 적고 정교한 튜닝이 가능하다면 **GAIL**이 가장 안전한 선택이다.
  - 구현의 단순함과 고성능을 동시에 원하며 데이터가 어느 정도 확보되었다면 **AdRIL**이 매력적인 대안이다.
- **한계점**: 본 연구는 AI가 생성한 전문가 데이터(D4RL)만을 사용하였다. 실제 인간의 시연 데이터(human demonstration)를 사용할 경우 하이퍼파라미터 최적화 방향이 달라질 수 있다는 점이 언급되었다.
- **비판적 해석**: 일부 알고리즘(DRIL, RED 등)이 Off-policy 설정에서 실패한 점은, 기존 논문들이 주장한 성능 향상이 특정 베이스 RL 알고리즘(On-policy)이나 특정 튜닝 조건에 의존적이었을 가능성을 제기한다.

## 📌 TL;DR

본 논문은 파편화되어 있던 심층 모방 학습(Deep IL) 알고리즘들을 SAC라는 공통의 Off-policy 프레임워크 위에서 재구현하고 공정하게 비교 분석하였다. 실험 결과, **GAIL**은 여전히 가장 강건한 성능을 보이며, **AdRIL**은 적은 튜닝으로 높은 효율을 내는 실용적인 대안임이 확인되었고, 데이터가 충분할 경우 **BC**만으로도 충분한 성능을 낼 수 있음을 입증하였다. 이 연구는 향후 IL 연구에서 공정한 벤치마크 설정의 중요성을 강조하며, 실무자들에게 데이터 양에 따른 알고리즘 선택 가이드를 제공한다.

# Online Meta-Critic Learning for Off-Policy Actor-Critic Methods

Wei Zhou, Yiying Li, Yongxin Yang, Huaimin Wang, Timothy M. Hospedales (2020)

## 🧩 Problem to Solve

본 논문은 Off-Policy Actor-Critic (OffP-AC) 방법론에서 Actor의 학습을 가속화하고 성능을 향상시키기 위해, 고정된 손실 함수 대신 학습 가능한 손실 함수를 도입하는 문제를 다룬다. 

일반적인 OffP-AC 방법론(DDPG, TD3, SAC 등)에서 Critic은 Temporal-Difference(TD) 학습을 통해 행동-가치 함수(Action-Value Function)를 추정하고, Actor는 이 Critic이 제공하는 손실 함수를 통해 기대 수익을 높이는 방향으로 업데이트된다. 그러나 이러한 손실 함수는 수동으로 설계되어 고정되어 있으며, 학습 과정 전반에 걸쳐 최적이 아닐 수 있다. 

기존의 Meta-learning 연구들은 다양한 작업군(Family of tasks)에 대해 오프라인으로 사전 학습을 수행한 후 새로운 작업에 적용하는 방식을 취함으로써 막대한 계산 비용이 발생하며, 많은 경우 On-policy 방식에 의존하여 샘플 효율성이 떨어진다는 한계가 있다. 따라서 본 논문의 목표는 단일 작업 내에서 온라인으로 학습 가능하며, 샘플 효율성이 높은 Off-policy 기반의 Meta-Critic 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Actor의 학습 과정을 관찰하여 학습 진척도를 최대화할 수 있는 추가적인 손실 함수를 생성하는 **Meta-Critic** 네트워크를 도입하는 것이다. 

주요 기여점은 다음과 같다.
1. **온라인 메타 학습**: 여러 작업군에 대한 사전 학습 없이, 단일 작업 내에서 Actor와 병렬적으로 Meta-Critic을 학습시킨다.
2. **Off-Policy 기반 설계**: 최신 OffP-AC 알고리즘(DDPG, TD3, SAC)에 유연하게 통합되어 높은 샘플 효율성을 유지한다.
3. **학습 진척도 최적화**: Meta-Critic은 단순히 가치 함수를 추정하는 것이 아니라, Actor가 검증 데이터(Validation data)에서 더 나은 성능을 내도록 유도하는 보조 손실 함수를 학습한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 차별점을 제시한다.

1. **Policy-Gradient (PG) 및 Off-Policy AC**: On-policy 방식은 매 업데이트마다 새로운 궤적(Trajectory)이 필요하여 비용이 많이 들지만, Off-policy 방식은 Replay Buffer를 통해 과거 경험을 재사용함으로써 샘플 효율성을 높인다.
2. **Meta-Learning for RL**: 기존의 Meta-RL은 빠른 적응 전략, 하이퍼파라미터, 혹은 내재적 보상(Intrinsic reward)을 학습한다. 그러나 대부분의 연구가 작업군 전체에 대한 오프라인 학습을 요구한다는 점에서 본 연구의 온라인 단일 작업 학습 방식과 차별화된다.
3. **Loss Learning**: 지도 학습에서의 Surrogate loss 학습이나 Meta-regularization 연구들이 존재한다. 본 연구는 이를 강화학습의 OffP-AC 구조에 맞게 설계하여 학습 속도를 높이는 데 집중한다.
4. **LIRPG**: 온라인으로 내재적 보상을 학습하는 LIRPG와 유사하지만, LIRPG는 단순히 스칼라 보상 값만 추가하여 Policy Gradient를 통해 간접적으로 영향을 주는 반면, Meta-Critic은 Actor 최적화를 위한 직접적인 손실 함수를 제공하여 훨씬 더 높은 샘플 효율성을 달성한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
본 프레임워크는 기존의 Critic 외에 보조적인 Meta-Critic 네트워크를 추가한다. Actor $\phi$는 기존 Critic이 제공하는 손실 $L_{critic}$과 Meta-Critic이 제공하는 손실 $L_{mcritic}_\omega$의 합을 최소화하는 방향으로 학습된다. Meta-Critic의 파라미터 $\omega$는 Actor가 학습된 후 검증 데이터에서 보이는 성능을 통해 최적화되는 Bi-level optimization 구조를 가진다.

### 상세 학습 절차 (Bi-level Optimisation)
학습 과정은 다음과 같은 두 단계의 최적화 루프로 구성된다.

1. **Lower-level Optimisation (Actor 학습)**:
   훈련 데이터 $d_{trn}$을 사용하여 Actor $\phi$를 업데이트한다.
   $$\phi_{new} = \phi - \eta \nabla_\phi L_{critic}(d_{trn}) - \eta \nabla_\phi L_{mcritic}_\omega(d_{trn})$$
   이때, 비교를 위해 Meta-Critic의 도움 없이 $L_{critic}$만으로 업데이트한 $\phi_{old}$도 함께 계산한다.

2. **Upper-level Optimisation (Meta-Critic 학습)**:
   검증 데이터 $d_{val}$을 사용하여 Meta-Critic $\omega$를 업데이트한다. Meta-loss는 Meta-Critic의 개입이 실제로 Actor의 성능을 향상시켰는지를 측정한다.

### 주요 방정식 및 손실 함수

**1. Meta-Loss 정의**
가장 직관적인 방법은 $\phi_{new}$의 검증 성능을 측정하는 것이나, 본 논문에서는 수치적 안정성을 위해 다음과 같은 Clipped difference 형태의 $L_{meta\_clip}$을 제안한다.
$$L_{meta\_clip} = \tanh(L_{critic}(d_{val}; \phi_{new}) - L_{critic}(d_{val}; \phi_{old}))$$
이 식은 Meta-Critic이 제공한 업데이트가 일반적인 학습보다 더 나은 결과를 냈는지를 평가하며, $\tanh$를 통해 그래디언트의 크기를 제한한다.

**2. Meta-Critic 아키텍처 ($h_\omega$)**
Meta-Critic은 Actor의 파라미터 $\phi$와 상태 $s$의 정보를 함께 활용해야 한다. 이를 위해 Actor 네트워크를 특성 추출기 $\bar{\pi}_\phi$와 의사 결정 모듈 $\hat{\pi}_\phi$로 분리하고, 특성 추출기의 출력(Penultimate layer)을 입력으로 사용하는 MLP 구조를 설계하였다. 

두 가지 설계 안이 제시되었다:
- **방법 1**: Actor-상태 공동 특성만을 사용
  $$h_\omega(d_{trn}; \phi) = \frac{1}{N} \sum_{i=1}^N f_\omega(\bar{\pi}_\phi(s_i))$$
- **방법 2**: 공동 특성에 상태 $s_i$와 행동 $a_i$를 추가로 입력
  $$h_\omega(d_{trn}; \phi) = \frac{1}{N} \sum_{i=1}^N f_\omega(\bar{\pi}_\phi(s_i), s_i, a_i)$$

최종 출력층에는 Softplus 활성화 함수를 사용하여 Meta-Critic이 제공하는 보조 손실이 항상 비음수(non-negative)가 되도록 하여, Vanilla Critic의 과대평가(Over-estimation) 문제를 완화하도록 설계하였다.

## 📊 Results

### 실험 설정
- **알고리즘**: DDPG, TD3, SAC를 베이스라인으로 하며, 여기에 Meta-Critic을 결합한 DDPG-MC, TD3-MC, SAC-MC를 제안한다.
- **데이터셋/작업**: OpenAI Gym의 MuJoCo V2 작업(HalfCheetah, Hopper, Walker2d, Ant 등), rllab의 MuJoCo 작업, 그리고 TORCS 시뮬레이션 레이싱.
- **평가 지표**: Max Average Return (최대 평균 보상).

### 주요 결과
1. **정량적 성능 향상**: 모든 OffP-AC 알고리즘에서 Meta-Critic을 추가했을 때 학습 속도가 빨라지고 최종 수렴 성능(Asymptotic performance)이 향상되었다. 특히 가장 도전적인 작업인 TORCS에서 SAC-MC가 뚜렷한 성능 향상을 보였다.
2. **안정성**: Meta-Critic을 적용한 모델들이 베이스라인 대비 보상의 분산(Variance)이 적어 더 안정적인 학습 곡선을 나타냈다.
3. **PPO-LIRPG와의 비교**: 온라인 메타 학습을 수행하는 PPO-LIRPG보다 Off-policy 기반의 Meta-Critic이 훨씬 더 높은 샘플 효율성과 성능을 보였다. 이는 단순히 보상 값을 수정하는 것보다 직접적인 손실 함수를 제공하는 것이 더 효과적임을 시사한다.
4. **제어 실험 (Control Experiments)**: 단순히 연산량(Gradient updates)을 늘리거나 파라미터 수를 늘리는 것만으로는 Meta-Critic의 성능 향상을 재현할 수 없었다. 이는 Meta-Critic이 단순한 자원 투입이 아닌, 더 나은 그래디언트 방향을 제시하고 있음을 증명한다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 Meta-Critic이 학습 초기 단계에서 Actor가 고보상 영역으로 더 빠르게 진입할 수 있도록 최적의 업데이트 방향을 정의해준다는 것을 보여주었다. 특히 가중치 공간에서의 최적화 궤적(Optimization trajectory) 시각화를 통해, Vanilla DDPG가 저보상 영역에서 방황하는 반면 DDPG-MC는 매우 직접적으로 고보상 영역으로 이동함을 확인하였다.

### 한계 및 향후 과제
1. **근시안적 최적화 (Myopic Optimization)**: 현재의 구조는 내부 루프(Base step)를 단 한 번만 수행하고 외부 루프(Meta step)를 업데이트하는 방식이다. 더 긴 시야(Longer horizon look-ahead)를 가진 최적화가 가능하지만, 이는 고계 도함수(Higher-order gradients) 계산으로 인한 메모리 증가와 그래디언트 불안정성 문제를 야기할 수 있다.
2. **단일 작업 한정**: 본 논문은 단일 작업에서의 온라인 학습에 집중하였으며, 향후에는 이를 다중 작업(Multi-task) 및 다중 도메인 RL로 확장할 계획이다.

## 📌 TL;DR

본 논문은 Off-Policy Actor-Critic 방법론의 고정된 손실 함수 문제를 해결하기 위해, 온라인으로 학습 가능한 **Meta-Critic** 프레임워크를 제안한다. Meta-Critic은 Actor의 학습 진척도를 극대화하는 보조 손실 함수를 생성하며, 이를 통해 DDPG, TD3, SAC와 같은 최신 알고리즘의 학습 속도와 최종 성능을 유의미하게 향상시킨다. 특히 단일 작업 내에서 온라인으로 학습되므로 기존 메타 학습의 막대한 비용 문제를 해결하였으며, 이는 향후 실물 로봇 제어와 같이 샘플 효율성이 극도로 중요한 분야에 적용될 가능성이 높다.
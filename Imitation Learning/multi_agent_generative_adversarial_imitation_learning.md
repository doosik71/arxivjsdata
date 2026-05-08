# Multi-Agent Generative Adversarial Imitation Learning

Jiaming Song, Hongyu Ren, Dorsa Sadigh, Stefano Ermon

## 🧩 Problem to Solve

기존 모방 학습(Imitation Learning, IL) 및 역강화 학습(Inverse Reinforcement Learning, IRL) 방법론은 단일 에이전트 환경에 주로 초점을 맞추고 있어, 여러 에이전트가 상호작용하는 다중 에이전트 설정(Markov games)에서는 여러 가지 한계에 직면합니다. 주요 문제점은 다음과 같습니다:

- **보상 함수 설계의 어려움**: 복잡한 다중 에이전트 시나리오(예: 다인용 게임, 다중 로봇 제어)에서 적절한 보상 함수를 수동으로 설계하기 매우 어렵습니다. 에이전트마다 보상 함수가 다를 수 있으며, 경쟁 환경에서는 상충될 수도 있습니다.
- **비정상적인 환경(Non-stationary environments)**: 다중 에이전트 환경에서 각 에이전트의 최적 정책은 다른 에이전트의 정책에 의존하므로, 개별 에이전트의 관점에서 환경이 비정상적으로 변할 수 있습니다.
- **다중 평형(Multiple equilibria) 문제**: 내쉬 균형(Nash equilibrium)과 같은 개념을 사용할 때 여러 해(solution)가 존재할 수 있어, 최적 정책을 찾는 것이 복잡해집니다.

## ✨ Key Contributions

이 논문은 다중 에이전트 모방 학습의 위와 같은 한계를 극복하기 위해 다음과 같은 핵심 기여를 합니다:

- **일반화된 다중 에이전트 IRL 프레임워크 제안**: 마르코프 게임(Markov games) 프레임워크를 기반으로 다중 에이전트 역강화 학습(MAIRL)의 새로운 개념과 공식화를 제시합니다. 이는 단일 에이전트 GAIL(Generative Adversarial Imitation Learning)을 엄격하게 일반화합니다.
- **MAGAIL(Multi-Agent GAIL) 알고리즘 개발**: 적대적 학습 방식을 사용하여 전문가 시연을 모방하는 실제적인 다중 에이전트 모방 학습 알고리즘인 MAGAIL을 제안합니다. 이는 협력적(centralized), 분산적(decentralized), 제로섬(zero-sum)의 세 가지 보상 구조에 대한 사전 지식(prior knowledge)을 통합할 수 있습니다.
- **MACK(Multi-agent Actor-Critic with Kronecker-factors) 알고리즘 도입**: 높은 분산(variance)을 가진 경사 추정 문제를 해결하고 샘플 효율성을 높이기 위해, MACK라는 다중 에이전트 액터-크리틱 알고리즘을 제안합니다. 이는 중앙 집중식 훈련-분산식 실행(centralized training with decentralized execution) 패러다임을 따릅니다.
- **다양한 고차원 환경에서의 우수한 성능 입증**: 입자 환경(particle environments) 및 협동 로봇 제어(cooperative robotic control)와 같은 고차원 환경에서 여러 협력적 또는 경쟁적 에이전트의 복잡한 행동을 성공적으로 모방함을 실험을 통해 보여줍니다.

## 📎 Related Works

- **단일 에이전트 모방 학습**:
  - **행동 복제(Behavior Cloning, BC)**: 전문가의 상태-행동 쌍을 사용하여 정책을 직접 학습하는 지도 학습 방식. 전이 확률이나 환경에 대한 지식이 필요 없지만, 누적 오류(compounding errors)와 공변량 변화(covariate shift) 문제에 취약합니다.
  - **역강화 학습(Inverse Reinforcement Learning, IRL)**: 전문가 정책이 미지의 보상 함수를 최적화한다고 가정하고, 해당 보상 함수를 복원한 후 강화 학습(RL)을 통해 정책을 학습하는 방식입니다. Maximum Entropy IRL (Ziebart et al., 2008)과 GAIL (Ho and Ermon, 2016) 등이 대표적입니다.
- **다중 에이전트 모방 학습**:
  - 대부분의 기존 연구는 에이전트들이 매우 구체적인 보상 구조를 가지고 있다고 가정합니다 (예: 완전 협력적 에이전트, 특정 역할 할당).
  - Lin et al. (2014)은 2인 제로섬 게임에 대한 IRL을 베이즈 추론으로 모델링했습니다.
  - Reddy et al. (2012)은 에이전트가 비협력적이지만 보상 함수가 미리 지정된 특징의 선형 조합이라고 가정했습니다.
- **다중 에이전트 강화 학습(MARL)**: Lowe et al. (2017)의 Multi-Agent Deep Deterministic Policy Gradient (MADDPG)와 Foerster et al. (2016)의 통신 학습 등이 있습니다. 본 연구는 이러한 최신 MARL 기법과 GAN 기반 생성 모델을 통합하는 일반적인 프레임워크를 제시합니다.

## 🛠️ Methodology

### 1. Markov 게임 및 Nash 균형에 대한 IRL 일반화

- **Markov Games 정의**: $N$개의 에이전트가 상태 $S$, 각 에이전트의 행동 집합 $\{A_i\}_{i=1}^N$, 전이 함수 $T$, 각 에이전트의 보상 함수 $r_i$를 공유하는 환경을 다룹니다. 각 에이전트는 자신의 예상 총 리턴 $R_i = \sum_{t=0}^\infty \gamma^t r_{i,t}$을 최대화합니다.
- **다중 에이전트 강화 학습(MARL) 목표**: 각 에이전트가 다른 에이전트의 정책에 대한 최적의 반응을 보이는 내쉬 균형 정책을 찾는 것을 목표로 하며, 모호성을 해결하기 위해 엔트로피 정규화($H(\pi)$)를 포함합니다.
  $$
  \text{MARL}(r) = \arg \min_{\pi \in \Pi, v \in \mathbb{R}^{S \times N}} f_r(\pi, v) - H(\pi)
  $$
- **다중 에이전트 역강화 학습(MAIRL) 공식화**: 단일 에이전트 IRL과 유사하게, 전문가 시연 $\pi^E$를 가장 잘 설명하는 보상 함수 $r$을 찾는 것을 목표로 합니다. 내쉬 균형 제약 조건을 시간차 학습(Temporal Difference learning) 기반의 동등한 제약 조건으로 변환하고, 라그랑주 이중(Lagrangian dual) 공식을 통해 이를 "이완(relax)"하여 전문가 정책과 다른 정책 간의 보상 차이를 나타냅니다.
  $$
  \text{MAIRL}_\psi(\pi^E) = \arg \max_r -\psi(r) + \sum_{i=1}^N (E_{\pi^E}[r_i]) - \left( \max_\pi \sum_{i=1}^N (\beta H_i(\pi_i) + E_{\pi_i, \pi^E_{-i}}[r_i]) \right)
  $$
  여기서 $\psi(r)$은 보상 함수 정규화자이며, $\beta$는 엔트로피 정규화 강도를 제어합니다.

### 2. MAGAIL(Multi-Agent Generative Adversarial Imitation Learning)

- **GAIL 확장**: $\psi$를 적대적 보상 함수 정규화자로 선택하여 GAIL을 다중 에이전트 설정으로 확장합니다. 각 에이전트 $i$는 자신만의 식별자(discriminator) $D_{\omega_i}$를 가지며, 이 식별자는 전문가 시연과 생성된 행동을 구별하도록 학습됩니다. 생성자는 식별자를 "속이기" 위해 보상을 최대화하도록 학습됩니다.
  $$
  \min_\theta \max_\omega E_{\pi_\theta} \left[ \sum_{i=1}^N \log D_{\omega_i}(s, a_i) \right] + E_{\pi^E} \left[ \sum_{i=1}^N \log(1 - D_{\omega_i}(s, a_i)) \right]
  $$
- **보상 구조에 따른 MAGAIL 유형**: 보상 구조에 대한 사전 지식을 통해 세 가지 MAGAIL 변형을 도입합니다.
  - **중앙 집중식(Centralized, Cooperative)**: 모든 에이전트가 동일한 보상 함수를 공유한다고 가정합니다 ($r_1 = r_2 = \dots = r_N$). 단일 식별자가 공동 정책의 행동을 평가합니다.
  - **분산식(Decentralized, Mixed)**: 각 에이전트가 고유한 보상 함수를 가진다고 가정합니다 ($r_i \in \mathbb{R}^{O_i \times A_i}$). 각 에이전트에 대해 별도의 식별자 $D_i$가 존재하며, 이는 해당 에이전트의 궤적을 구별합니다.
  - **제로섬(Zero-Sum, Competitive)**: 두 에이전트가 반대되는 보상 함수를 갖는다고 가정합니다 ($r_1 = -r_2$). 식별자는 한 에이전트의 보상을 최대화하고 다른 에이전트의 보상을 최소화하는 방식으로 학습됩니다.

### 3. MACK(Multi-agent Actor-Critic with Kronecker-factors)

- **생성자(Generator) 최적화**: MAGAIL의 정책 $\pi_\theta$를 최적화하기 위해, ACKTR(Actor-Critic with Kronecker-factored Trust Region)에서 영감을 받은 MACK를 사용합니다.
- **중앙 집중식 훈련, 분산식 실행**: 훈련 시에는 모든 에이전트의 관측 및 행동을 사용하여 분산 감소를 위한 가치 함수 $V_{\phi_i}(s, a_{-i})$를 계산하지만, 실행 시에는 이 추가 정보($a_{-i}$)를 사용하지 않습니다.
- **자연 정책 경사(Natural Policy Gradient)**: K-FAC(Kronecker-factored Approximate Curvature)를 사용하여 자연 정책 경사를 효율적으로 근사하고, 학습률 스케줄링을 통해 안정적인 훈련을 달성합니다.

## 📊 Results

### 1. 입자 환경 (Particle Environments)

- **협력적 과제 (Cooperative Communication, Cooperative Navigation)**:
  - MAGAIL (중앙 집중식, 분산식)은 행동 복제(BC) 및 개별 GAIL보다 지속적으로 우수한 성능을 보였습니다.
  - 특히, 중앙 집중식 MAGAIL은 200회 시연만으로 전문가 수준의 성능을 달성했지만, BC는 400회 시연으로도 근접하지 못했습니다.
  - 사전 지식(중앙 집중식 설정)이 없는 분산식 MAGAIL도 두 에이전트 간에 높은 상관 관계를 가진 보상을 학습하여 좋은 성능을 보였습니다.
- **경쟁적 과제 (Keep-Away, Predator-Prey)**:
  - 분산식 및 제로섬 MAGAIL은 종종 중앙 집중식 MAGAIL과 BC보다 우수한 성능을 보였습니다. 이는 보상 구조에 대한 적절한 사전 지식($\hat{\psi}$) 선택이 중요하다는 것을 시사합니다.

### 2. 협동 제어 (Cooperative Control)

- **비이상적인 전문가 시연에 대한 적응**: 2족 보행 로봇들이 긴 판자를 함께 옮기는 과제에서, 환경이 변화하여 전문가 시연이 더 이상 최적이 아닐 때를 가정했습니다. (예: 지면에 범프가 있고 판자가 더 가벼워져 과도한 힘을 사용하기 쉬운 환경).
  - BC로 훈련된 에이전트는 공격적으로 행동하여 실패율이 39.8% (보상 1.26)에 달했습니다.
  - 중앙 집중식 MAGAIL로 훈련된 에이전트는 새로운 환경에 적응하여 실패율이 26.2% (보상 26.57)로 현저히 낮았습니다.

## 🧠 Insights & Discussion

- **다중 에이전트 IRL 프레임워크의 중요성**: 이 연구는 다중 에이전트 환경에서 보상 함수를 명시적으로 설계하지 않고도 복잡한 협력적 및 경쟁적 행동을 모방할 수 있는 일반적인 프레임워크를 제공합니다. 이는 실제 다중 에이전트 시스템 배포에 중요한 진전입니다.
- **보상 구조 사전 지식의 활용**: MAGAIL은 환경의 보상 구조(협력적, 경쟁적, 혼합)에 대한 사전 지식을 식별자 설계에 통합하여 성능을 향상시킬 수 있음을 보여주었습니다. 이는 문제 특성에 맞는 모델 선택의 중요성을 강조합니다.
- **높은 샘플 효율성 및 안정적인 학습**: MACK와 같은 액터-크리틱 방식을 활용하여 강화 학습의 고질적인 문제인 높은 분산 경사 추정을 완화하고, 샘플 효율적인 학습을 가능하게 합니다.
- **제한 사항 및 향후 연구**:
  - MARL($r$)이 고유한 해를 가진다는 가정은 항상 참이 아닐 수 있습니다.
  - 전문가 정책 $\pi^E$에 직접 접근할 수 없다는 가정을 확장하여, 전문가가 학습 과정의 일부에 참여하여 에이전트를 돕는 협력적 역강화 학습(CIRL)과 같은 시나리오를 탐색하는 것이 흥미로운 연구 방향이 될 수 있습니다.
  - 다양한 형태의 전문가 시연(예: 시각적 시연)으로부터의 학습 및 해석 가능한 모방 학습에 대한 연구가 추가적으로 필요합니다.

## 📌 TL;DR

이 논문은 다중 에이전트 환경에서 복잡한 행동을 모방하기 위한 새로운 역강화 학습 프레임워크인 **MAGAIL(Multi-Agent Generative Adversarial Imitation Learning)**을 제안합니다. 기존 단일 에이전트 GAIL을 내쉬 균형 개념을 통해 다중 에이전트 설정으로 일반화하며, 협력적, 분산적, 제로섬 등 다양한 보상 구조에 대한 사전 지식을 활용할 수 있도록 세 가지 변형을 제시합니다. 생성자(정책) 최적화를 위해 K-FAC 기반의 다중 에이전트 액터-크리틱 알고리즘인 **MACK**를 도입하여 높은 분산 경사 추정 문제를 해결하고 샘플 효율성을 높입니다. 실험 결과, MAGAIL은 입자 환경 및 협동 로봇 제어와 같은 고차원 환경에서 BC나 개별 GAIL보다 우수한 전문가 행동 모방 성능을 보이며, 특히 비최적 전문가 시연에도 적응하는 능력을 입증했습니다.

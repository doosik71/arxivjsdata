# A Brief Survey of Deep Reinforcement Learning

Kai Arulkumaran, Marc Peter Deisenroth, Miles Brundage, Anil Anthony Bharath (2017)

## 🧩 Problem to Solve

본 논문은 강화 학습(Reinforcement Learning, RL)이 직면해 온 근본적인 한계인 확장성(Scalability) 문제를 해결하고자 하는 Deep Reinforcement Learning(DRL)의 전반적인 흐름을 분석한다. 전통적인 RL은 상태(State)와 행동(Action) 공간이 저차원인 문제에서는 성공적이었으나, 고차원 데이터(예: 이미지 픽셀)를 다루어야 하는 환경에서는 메모리 복잡도, 계산 복잡도, 그리고 샘플 복잡도(Sample Complexity)의 문제로 인해 적용이 불가능했다. 즉, 소위 '차원의 저주(Curse of Dimensionality)'로 인해 실제 세계의 복잡한 시각적 환경에서 동작하는 자율 시스템을 구축하는 것이 매우 어려웠다. 따라서 본 논문의 목표는 딥러닝의 표현 학습(Representation Learning) 능력을 RL에 결합하여, 고차원 입력으로부터 직접 최적의 행동 정책을 학습하는 DRL의 핵심 알고리즘과 방법론을 체계적으로 정리하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 직관은 심층 신경망(Deep Neural Networks)이 고차원 데이터로부터 콤팩트한 저차원 특징(Feature)을 자동으로 추출할 수 있다는 점을 RL의 의사결정 과정에 도입하는 것이다. 이를 통해 RL 에이전트가 환경의 원시 입력(Raw Input, 예: 이미지)으로부터 직접 유의미한 표현을 학습하고, 이를 바탕으로 가치 함수를 근사하거나 정책을 최적화함으로써 과거에는 불가능했던 고차원 문제(예: Atari 게임, 로봇 제어)를 해결할 수 있게 되었음을 보여준다.

## 📎 Related Works

논문에서는 RL의 기초가 되는 Markov Decision Process(MDP)와 이를 확장한 Partially Observable MDP(POMDP)를 소개한다. 기존의 RL 접근 방식은 주로 정형화된 표(Tabular) 방식이나 단순한 비매개변수적 방법에 의존했으나, 이는 상태 공간이 조금만 커져도 메모리 요구량이 기하급수적으로 증가하는 한계가 있었다. 또한, 전통적인 제어 이론(Optimal Control)은 환경의 전이 역학(Transition Dynamics) 모델이 주어져야 하지만, RL은 모델 없이 시행착오(Trial and Error)를 통해 학습한다는 점에서 차별화된다. 최근의 DRL은 이러한 전통적 RL의 수학적 프레임워크 위에 심층 신경망의 강력한 함수 근사(Function Approximation) 능력을 더함으로써 기존의 한계를 극복하고 있다.

## 🛠️ Methodology

### 1. 기본 프레임워크: MDP

RL은 다음과 같은 MDP 구성 요소로 정의된다.

- 상태 집합 $S$와 시작 상태 분포 $p(s_0)$
- 행동 집합 $A$
- 전이 역학(Transition Dynamics) $T(s_{t+1}|s_t, a_t)$
- 즉각 보상 함수(Reward Function) $R(s_t, a_t, s_{t+1})$
- 할인 계수(Discount Factor) $\gamma \in [0, 1]$

에이전트의 목표는 기대 누적 보상(Expected Return)을 최대화하는 최적 정책 $\pi^*$를 찾는 것이며, 이는 다음과 같은 식으로 표현된다.
$$\pi^* = \text{argmax}_\pi E[R|\pi]$$

### 2. 가치 기반 방법 (Value-Based Methods)

상태 가치 함수 $V^\pi(s)$와 상태-행동 가치 함수 $Q^\pi(s, a)$를 학습하여 최적의 행동을 결정한다.

- **Bellman Equation**: $Q$ 함수는 다음과 같은 재귀적 형태로 업데이트된다.
$$Q^\pi(s_t, a_t) = E_{s_{t+1}}[r_{t+1} + \gamma Q^\pi(s_{t+1}, \pi(s_{t+1}))]$$
- **Deep Q-Network (DQN)**: CNN을 통해 이미지로부터 특징을 추출하고 $Q$ 값을 예측한다. 학습의 불안정성을 해결하기 위해 두 가지 핵심 기술을 사용한다.
  - **Experience Replay**: 경험 $(s_t, a_t, s_{t+1}, r_{t+1})$을 버퍼에 저장하고 무작위로 샘플링하여 학습함으로써 데이터 간의 시간적 상관관계를 끊고 학습 효율을 높인다.
  - **Target Networks**: $Q$ 값 업데이트를 위한 타겟 값을 계산하는 별도의 네트워크를 두어, 타겟이 계속 변함으로 인해 발생하는 진동 현상을 방지하고 학습을 안정화한다.

### 3. 정책 탐색 방법 (Policy Search Methods)

가치 함수를 거치지 않고 정책 $\pi_\theta$를 직접 최적화한다.

- **Policy Gradient**: 보상을 최대화하는 방향으로 파라미터 $\theta$를 업데이트한다. 이때 REINFORCE 규칙(Score Function Estimator)을 사용하여 기울기를 추정한다.
$$\nabla_\theta E_X[f(X; \theta)] = E_X[f(X; \theta) \nabla_\theta \log p(X)]$$
- **Trust Region Policy Optimization (TRPO)**: 업데이트 시 정책이 급격하게 변하는 것을 막기 위해 KL Divergence를 이용해 업데이트 범위를 제한하는 Trust Region을 설정한다.

### 4. Actor-Critic 방법

가치 함수(Critic)와 정책(Actor)을 동시에 학습하는 하이브리드 방식이다.

- **Actor**: 현재 상태에서 행동을 선택하며, Critic의 피드백을 받아 정책을 업데이트한다.
- **Critic**: 현재 정책의 가치를 평가하여 TD-Error를 계산하고, 이를 Actor에게 전달하여 분산을 줄인 효율적인 학습을 가능하게 한다.
- **A3C (Asynchronous Advantage Actor-Critic)**: 여러 개의 에이전트를 병렬로 실행하여 각자 환경과 상호작용하며 비동기적으로 전역 네트워크를 업데이트함으로써 학습 속도와 안정성을 높인다.

## 📊 Results

본 논문은 DRL의 성능을 입증하기 위해 사용된 다양한 벤치마크와 실제 적용 사례를 설명한다.

- **Atari 2600 (ALE)**: DQN은 픽셀 입력만으로 여러 Atari 게임에서 인간 수준의 성능을 달성하였으며, 이는 DRL이 고차원 시각 데이터를 처리할 수 있음을 증명한 사례이다.
- **Board Games**: AlphaGo는 지도 학습(Supervised Learning)과 강화 학습, 그리고 휴리스틱 탐색 알고리즘을 결합하여 세계 챔피언을 꺾는 성과를 냈다.
- **Continuous Control**: MuJoCo 물리 엔진을 이용한 실험을 통해, TRPO, PPO, DDPG 등의 알고리즘이 로봇의 관절 제어와 같은 연속적인 행동 공간에서도 효과적으로 동작함을 확인하였다.
- **Robotics**: 카메라의 RGB 픽셀 입력에서 로봇의 토크 출력으로 직접 연결되는 End-to-End visuomotor policies 학습이 가능함을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 의의

DRL의 가장 큰 강점은 심층 신경망의 표현 학습 능력을 통해 RL의 고질적인 문제였던 '차원의 저주'를 극복했다는 점이다. 특히 Experience Replay와 Target Network 같은 기법들은 함수 근사(Function Approximation)와 RL의 결합 시 발생하는 불안정성 문제를 효과적으로 해결하였다.

### 한계 및 과제

- **샘플 효율성(Sample Efficiency)**: 여전히 많은 양의 상호작용 데이터가 필요하며, 실제 로봇 환경에서는 하드웨어 마모 및 시간 비용 문제로 인해 직접 학습이 어렵다.
- **탐색과 이용의 딜레마(Exploration vs. Exploitation)**: $\epsilon$-greedy와 같은 단순한 방식 외에, 불확실성을 기반으로 한 효율적인 탐색 전략(예: UCB, Intrinsic Motivation)이 더 필요하다.
- **일반화 및 전이 학습**: 특정 환경에서 학습된 정책을 새로운 환경으로 전이하는 능력이 부족하며, 이를 해결하기 위한 Sim-to-Real 전이 학습 연구가 필수적이다.

### 비판적 해석

본 논문은 광범위한 알고리즘을 잘 요약하고 있으나, 이론적 분석보다는 기존 알고리즘의 나열과 사례 제시 중심의 서베이 성격을 띤다. 특히, 신경망의 블랙박스 특성으로 인해 RL의 수렴성 보장이 어렵다는 점에 대해 더 깊은 이론적 논의가 추가되었다면 더 완성도 높은 보고서가 되었을 것이다.

## 📌 TL;DR

본 논문은 고차원 데이터 처리 능력을 가진 딥러닝과 최적 행동을 찾는 강화 학습을 결합한 **Deep Reinforcement Learning(DRL)**의 전반적인 방법론을 다룬 서베이 논문이다. 가치 기반(DQN), 정책 기반(TRPO, PPO), 그리고 이 둘을 결합한 Actor-Critic(A3C) 구조를 중심으로 DRL의 발전 과정을 정리하였으며, 이를 통해 Atari 게임, 보드게임, 로봇 제어 등에서 거둔 성과를 분석한다. 이 연구는 향후 일반 인공지능(AGI) 구축을 위해 시각적 이해와 자율적 의사결정을 통합하는 핵심적인 가이드라인을 제공한다.

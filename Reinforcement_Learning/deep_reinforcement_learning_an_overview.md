# DEEP REINFORCEMENT LEARNING: AN OVERVIEW

Yuxi Li (2018)

## 🧩 Problem to Solve

본 논문은 최근 급격하게 발전하고 있는 Deep Reinforcement Learning (Deep RL) 분야의 방대한 성과들을 체계적으로 정리하고 분석하는 것을 목표로 한다. Reinforcement Learning (RL)은 에이전트가 환경과 상호작용하며 시행착오를 통해 최적의 정책을 학습하는 순차적 의사결정 문제(Sequential Decision Making Problem)를 다룬다.

기존의 RL은 상태 공간이 커질수록 발생하는 '차원의 저주(Curse of Dimensionality)' 문제로 인해 수동적인 Feature Engineering에 의존해야 했으며, 이는 많은 시간과 도메인 지식을 요구하는 한계가 있었다. Deep Learning의 등장으로 자동화된 표현 학습(Representation Learning)이 가능해지면서 RL의 성능이 비약적으로 향상되었으나, 이 과정에서 발생하는 학습의 불안정성과 이론적 공백이 존재한다. 따라서 본 논문은 Deep RL의 핵심 요소, 주요 메커니즘, 그리고 다양한 적용 사례를 종합적으로 분석하여 연구자들에게 학술적인 가이드라인을 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 Deep RL이라는 광범위한 분야를 **6가지 핵심 요소(Core Elements), 6가지 중요 메커니즘(Important Mechanisms), 12가지 응용 분야(Applications)**라는 체계적인 구조로 분류하여 정리했다는 점이다.

중심적인 직관은 Deep Learning의 강력한 표현 능력과 RL의 의사결정 프레임워크를 결합함으로써, 도메인 지식에 대한 의존도를 낮추고 End-to-End 학습을 통해 복잡한 환경에서도 최적의 정책을 찾을 수 있다는 것이다. 특히, Off-policy 학습, Function Approximation, Bootstrapping이 결합될 때 발생하는 불안정성(이른바 'Deadly Triad' 문제)을 DQN과 같은 알고리즘들이 어떻게 해결하여 학습을 안정화시켰는지를 분석한 점이 핵심이다.

## 📎 Related Works

논문은 Machine Learning, Deep Learning, 그리고 Reinforcement Learning의 기초 이론을 먼저 소개한다. 특히 Sutton과 Barto의 RL 기초 이론을 바탕으로, 가치 함수(Value Function), 정책 최적화(Policy Optimization), 그리고 시간차 학습(Temporal Difference Learning) 등의 전통적인 접근 방식을 다룬다.

기존의 RL 방식들은 Tabular 형태의 저장 공간을 사용했기에 상태 공간이 매우 큰 문제에 적용하기 어려웠다. 이를 해결하기 위해 Linear Function Approximation 등이 시도되었으나, 비선형 함수 근사기인 Deep Neural Network를 도입했을 때 발생하는 발산(Divergence) 문제가 주요 한계로 지적되었다. 본 논문은 이러한 한계를 극복하고 등장한 DQN(Deep Q-Network)과 AlphaGo 등의 최신 연구들을 기존 연구의 연장선에서 분석하며 차별점을 제시한다.

## 🛠️ Methodology

본 논문은 특정 알고리즘 하나를 제안하는 것이 아니라, Deep RL의 전체 파이프라인과 주요 방법론들을 분석한다. 주요 내용은 다음과 같다.

### 1. 가치 기반 방법론 (Value-Based Methods)

가장 대표적인 DQN은 Q-learning을 심층 신경망으로 확장한 것으로, 다음과 같은 핵심 기법을 통해 학습을 안정화한다.

- **Experience Replay**: 에이전트의 경험 $(s, a, r, s')$을 리플레이 메모리에 저장하고 무작위로 샘플링하여 학습함으로써, 데이터 간의 상관관계를 줄이고 학습 효율을 높인다.
- **Target Network**: 타겟 값을 계산하는 네트워크를 별도로 분리하여 주기적으로 업데이트함으로써, 학습 목표값이 계속 변하여 발생하는 진동 현상을 방지한다.
- **손실 함수**: DQN의 목표는 다음과 같은 MSE(Mean Squared Error) 손실 함수를 최소화하는 것이다.
$$L(\theta) = \mathbb{E} \left[ (y_j - Q(\phi_j, a_j; \theta))^2 \right]$$
여기서 타겟 $y_j$는 다음과 같이 정의된다.
$$y_j = r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)$$

### 2. 정책 기반 및 하이브리드 방법론 (Policy-Based & Hybrid Methods)

- **Policy Gradient**: 가치 함수를 거치지 않고 정책 $\pi(a|s; \theta)$를 직접 최적화한다. REINFORCE 알고리즘은 기대 보상의 기울기를 따라 $\theta$를 업데이트한다.
- **Actor-Critic**: 정책을 결정하는 Actor와 가치 함수를 평가하는 Critic이 공존한다. Critic이 제공하는 Advantage 값 $A(s, a) = Q(s, a) - V(s)$을 사용하여 Actor의 정책을 업데이트함으로써 분산을 줄이고 학습 속도를 높인다.
- **A3C (Asynchronous Advantage Actor-Critic)**: 여러 개의 에이전트를 병렬로 실행하여 서로 다른 경험을 쌓게 함으로써 Experience Replay 없이도 학습을 안정화하고 속도를 획기적으로 향상시킨다.

### 3. AlphaGo 및 AlphaGo Zero의 구조

AlphaGo Zero는 인간의 데이터 없이 Self-play만으로 학습하며, 다음과 같은 반복적인 정책 개선(Policy Iteration) 과정을 거친다.

- **MCTS (Monte Carlo Tree Search)**: 현재 상태에서 미래의 수를 탐색하여 더 나은 이동 확률을 계산하는 정책 개선 연산자로 작동한다.
- **Neural Network**: 정책 $\pi$와 가치 $v$를 동시에 예측하는 단일 네트워크를 사용한다.
- **손실 함수**: 가치 예측 오차와 정책 유사도(Cross-entropy)를 동시에 최적화한다.
$$l = (z-v)^2 - \pi^T \log p + c\|\theta\|^2$$

## 📊 Results

본 논문은 다양한 벤치마크와 적용 사례를 통해 Deep RL의 성과를 정량적/정성적으로 설명한다.

- **게임 (Games)**: DQN은 49가지 Atari 게임에서 인간 전문가 수준의 성능을 보였으며, AlphaGo Zero는 인간의 개입 없이 스스로 학습하여 세계 최고의 기사들을 능가하는 초인적인(Superhuman) 수준에 도달하였다.
- **로보틱스 (Robotics)**: Guided Policy Search (GPS)를 통해 복잡한 고차원 조작 작업에서 Raw Pixel 입력만으로 직접적인 토크 제어가 가능함을 입증하였다.
- **자연어 처리 (NLP)**: 기계 번역에서 Dual Learning 메커니즘을 적용하여 병렬 코퍼스 데이터가 부족한 상황에서도 성능을 크게 향상시켰으며, 대화 시스템에서 RL을 통한 정책 학습으로 사용자 만족도를 높였다.
- **기타 분야**: 금융의 포트폴리오 최적화, 헬스케어의 개인 맞춤형 치료 전략(DTRs), 컴퓨터 시스템의 리소스 할당 최적화 등에서 기존의 휴리스틱 방식보다 뛰어난 효율성을 보였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

Deep RL의 가장 큰 강점은 **자동화된 표현 학습**에 있다. 과거에는 전문가가 수동으로 특징을 추출해야 했으나, Deep RL은 원시 데이터(Raw Data)로부터 직접 유용한 특징을 학습함으로써 도메인 지식에 대한 의존도를 획기적으로 낮추었다. 또한, MCTS와 같은 탐색 알고리즘과 신경망의 결합이 거대한 탐색 공간을 가진 문제(예: 바둑)를 해결하는 결정적인 열쇠가 되었음을 보여준다.

### 한계 및 비판적 해석

- **블랙박스 문제**: 심층 신경망의 내부 동작 원리를 명확히 설명할 수 없는 해석 가능성(Interpretability)의 부재가 여전한 문제로 남아 있다.
- **Sim-to-Real Gap**: 시뮬레이션 환경(예: Atari, Go)에서의 성공이 실제 물리 환경(로보틱스, 자율주행)으로 직접 전이되지 않는다. 실제 환경에서는 완벽한 모델을 얻기 어렵고 데이터 수집 비용이 매우 높기 때문이다.
- **데이터 효율성**: 여전히 수백만 번의 상호작용이 필요하며, 이는 실제 세계의 적용에 있어 큰 진입장벽이 된다.

## 📌 TL;DR

본 논문은 Deep RL의 방대한 연구 흐름을 핵심 요소, 메커니즘, 응용 분야로 나누어 체계화한 종합 보고서이다. Deep Learning의 표현 능력과 RL의 의사결정 능력을 결합하여 Atari 게임, 바둑, 로보틱스 등에서 획기적인 성과를 거두었음을 분석하며, 특히 DQN, A3C, AlphaGo Zero와 같은 핵심 알고리즘의 작동 원리와 안정화 기법을 상세히 다룬다. 이 연구는 향후 AGI(인공 일반 지능) 구현을 위해 필수적인 기반 지식을 제공하며, 시뮬레이션과 현실 세계 사이의 간극을 줄이는 것이 다음 세대의 핵심 과제임을 시사한다.

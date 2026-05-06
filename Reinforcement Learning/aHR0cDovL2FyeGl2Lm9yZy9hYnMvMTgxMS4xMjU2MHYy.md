# An Introduction to Deep Reinforcement Learning

Vincent François-Lavet, Peter Henderson, Riashat Islam, Marc G. Bellemare and Joelle Pineau (2018)

## 🧩 Problem to Solve

본 논문(또는 기술 보고서)은 복잡한 환경에서의 순차적 의사결정(Sequential Decision-Making) 문제를 해결하기 위한 Deep Reinforcement Learning(Deep RL)의 전반적인 이론과 방법론을 체계적으로 정리하는 것을 목표로 한다.

전통적인 강화학습(RL)은 상태 공간이 커질수록 특징 설계(Feature Engineering)에 과도하게 의존해야 하며, 고차원 입력 데이터(예: 이미지 픽셀)를 직접 처리하는 데 한계가 있었다. 이러한 '차원의 저주' 문제를 해결하기 위해 딥러닝(Deep Learning)의 함수 근사(Function Approximation) 능력을 강화학습에 결합함으로써, 사람이 직접 특징을 설계하지 않고도 데이터로부터 추상적인 표현을 직접 학습하여 복잡한 제어 문제를 해결하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 딥러닝과 강화학습의 결합을 통한 고차원 상태 공간의 제어 가능성을 제시하고, 이를 위한 알고리즘 체계를 다음과 같이 구조화하여 제공한 점이다.

1. **심층 강화학습 방법론의 체계적 분류**: Model-free 방식(Value-based 및 Policy-based)과 Model-based 방식으로 나누어 각각의 핵심 알고리즘과 발전 과정을 상세히 분석한다.
2. **일반화(Generalization) 개념의 심층 분석**: RL에서의 일반화를 Sample Efficiency 및 Transfer Learning 관점에서 정의하고, Bias-Overfitting Trade-off 관점에서 이를 최적화하는 방법을 제시한다.
3. **실무적 벤치마킹 가이드라인 제공**: 단순한 성능 수치 제시를 넘어, 무작위 시드(Random Seed)의 영향, 통계적 유의성 검정, 하이퍼파라미터 튜닝 등 실험의 재현성(Reproducibility)을 위한 최적 실무(Best Practices)를 제안한다.
4. **MDP를 넘어선 확장 가능성 논의**: POMDP, Meta-learning, Multi-agent system 등 실제 환경에서 마주하게 되는 비마르코프(Non-Markovian) 설정에 대한 해결책을 다룬다.

## 📎 Related Works

본 논문은 기존의 강화학습 기초 이론(Sutton & Barto 등)과 최신 딥러닝 성과들을 연결한다.

* **기존 접근 방식의 한계**: 전통적인 RL은 Tabular 방식이나 선형 함수 근사에 의존하여, 고차원 입력 데이터에서 상태를 구분하기 위한 정교한 Feature Selection이 필수적이었다.
* **차별점**: Deep RL은 합성곱 신경망(CNN)이나 순환 신경망(RNN)과 같은 심층 구조를 통해 저수준 데이터(픽셀 등)로부터 고수준의 추상적 특징을 자동으로 추출한다. 이를 통해 Atari 게임, 바둑(Go), 포커 등 과거에는 불가능하다고 여겨졌던 복잡한 환경에서 초인적인 성능(Super-human level)을 달성할 수 있게 되었다.

## 🛠️ Methodology

본 보고서에서는 Deep RL의 세 가지 주요 접근 방식을 중심으로 방법론을 설명한다.

### 1. 기본 프레임워크: MDP (Markov Decision Process)

강화학습은 5-tuple $(S, A, T, R, \gamma)$로 정의되는 MDP를 기반으로 한다. 에이전트는 상태 $s \in S$에서 행동 $a \in A$를 취하며, 전이 함수 $T$에 의해 다음 상태 $s'$로 이동하고 보상 $R$을 얻는다. 목표는 누적 기대 보상(Expected Return) $V^\pi(s)$를 최대화하는 최적 정책 $\pi^*$를 찾는 것이다.

### 2. Value-based Methods

가치 함수를 학습하여 이를 통해 정책을 유도하는 방식이다.

* **Q-learning**: 벨만 방정식(Bellman Equation)을 이용하여 최적 가치 함수 $Q^*$를 학습한다.
    $$Q^*(s,a) = \sum_{s' \in S} T(s,a,s') (R(s,a,s') + \gamma \max_{a' \in A} Q^*(s', a'))$$
* **DQN (Deep Q-Network)**: 신경망을 가치 함수 근사기로 사용하며, 불안정성을 줄이기 위해 다음 두 가지 핵심 기법을 도입한다.
  * **Experience Replay**: 과거의 경험 $\langle s, a, r, s' \rangle$을 버퍼에 저장하고 무작위로 샘플링하여 학습함으로써 데이터 간 상관관계를 줄인다.
  * **Target Network**: 타겟 값 계산에 사용되는 네트워크 파라미터를 주기적으로만 업데이트하여 학습 목표의 변동성을 줄인다.
* **개선된 변형들**:
  * **Double DQN**: 행동 선택과 가치 평가를 분리하여 Q-value의 과대평가(Overestimation) 편향을 제거한다.
  * **Dueling Network**: $Q$ 함수를 상태 가치 $V(s)$와 이득 함수 $A(s,a)$로 분리하여 학습 효율을 높인다.
  * **Distributional DQN**: 단일 기대값이 아닌 보상의 확률 분포 $Z^\pi$를 학습하여 더 풍부한 학습 신호를 확보한다.

### 3. Policy Gradient Methods

가치 함수를 거치지 않고 정책 $\pi_w$를 직접 최적화하는 방식이다.

* **Stochastic Policy Gradient**: 정책 경사 정리(Policy Gradient Theorem)에 따라 다음과 같이 파라미터 $w$를 업데이트한다.
    $$\nabla_w V^{\pi_w}(s_0) = \mathbb{E}_{s \sim \rho, a \sim \pi} [\nabla_w \log \pi_w(s,a) Q^{\pi_w}(s,a)]$$
* **Actor-Critic**: 정책을 결정하는 Actor와 가치 함수를 평가하는 Critic을 동시에 학습시켜, 분산을 줄이고 학습 속도를 높인다.
* **TRPO & PPO**: 정책 업데이트 시 이전 정책과의 KL Divergence를 제한하거나(TRPO), 목적 함수에 Clipping 기법을 적용하여(PPO) 급격한 정책 변화로 인한 성능 저하를 방지한다.

### 4. Model-based Methods

환경의 전이 함수 $T$와 보상 함수 $R$을 학습하여 내부 모델을 구축하고, 이를 이용해 Planning을 수행하는 방식이다. 대표적으로 Monte-Carlo Tree Search(MCTS)나 Trajectory Optimization이 사용된다.

## 📊 Results

본 논문은 특정 실험 결과보다는 기존 연구들의 벤치마크 성과를 종합적으로 제시한다.

* **데이터셋 및 작업**: Atari 2600 게임(ALE), 바둑(Go), 포커, MuJoCo 로봇 제어 시뮬레이션 등이 주요 벤치마크로 사용되었다.
* **주요 성과**:
  * **Atari**: DQN 및 그 변형 알고리즘들이 픽셀 입력만으로 다수의 게임에서 인간 수준 이상의 성능을 달성하였다.
  * **보드게임**: AlphaGo와 같은 시스템이 MCTS와 Deep RL의 결합을 통해 바둑에서 세계 챔피언을 꺾는 성과를 냈다.
  * **로봇 제어**: DDPG, PPO 등이 MuJoCo 환경에서 연속적인 행동 공간(Continuous Action Space)을 효과적으로 제어함을 입증하였다.
* **실험적 시사점**: 단순한 평균 성능 보고보다 통계적 유의성 검정과 에이블레이션 연구(Ablation Study)가 필수적임을 강조하며, 특히 하이퍼파라미터 설정에 따라 결과가 크게 달라질 수 있음을 경고한다.

## 🧠 Insights & Discussion

### 강점 및 가능성

Deep RL은 고차원 데이터로부터의 자동 특징 추출 능력을 통해, 인간의 개입 없이도 복잡한 환경에서의 최적 행동을 학습할 수 있음을 보여주었다. 특히 Model-free와 Model-based의 결합은 샘플 효율성(Sample Efficiency)과 계산 효율성 사이의 균형을 맞출 수 있는 유망한 방향이다.

### 한계 및 해결 과제

1. **Reality Gap (Sim-to-Real)**: 시뮬레이션에서 학습한 정책이 실제 물리 환경의 미세한 차이로 인해 작동하지 않는 문제가 존재한다. 이를 위해 Domain Randomization이나 Transfer Learning 기법이 요구된다.
2. **Sample Efficiency**: 특히 Model-free 방식은 최적 정책을 찾기 위해 방대한 양의 상호작용 데이터가 필요하며, 이는 실제 환경(의료, 금융 등)에서 적용하기 어렵게 만든다.
3. **Exploration-Exploitation Dilemma**: 보상이 희소한(Sparse Reward) 환경에서는 단순한 $\epsilon$-greedy 방식으로는 효율적인 탐색이 불가능하며, Intrinsic Motivation(내재적 동기)이나 Curiosity-driven exploration 등의 고도화된 전략이 필요하다.

### 비판적 해석

본 논문은 Deep RL의 광범위한 기술적 스펙트럼을 매우 훌륭하게 정리하고 있으나, 개별 알고리즘의 수렴성(Convergence)에 대한 엄밀한 이론적 증명보다는 경험적 성능 향상에 치중한 경향이 있다. 또한, 신경망의 거대한 파라미터 수로 인해 발생하는 과적합(Overfitting) 문제를 일반화 관점에서 다루고 있으나, 실제 복잡한 환경에서 이를 완벽히 제어하기 위한 구체적인 정규화 방안은 여전히 연구 대상이다.

## 📌 TL;DR

본 논문은 강화학습과 딥러닝의 결합인 **Deep RL의 핵심 이론, 알고리즘(Value-based, Policy-based, Model-based), 그리고 실무적 적용 방안을 집대성한 종합 가이드라인**이다. 고차원 입력 데이터를 처리하는 신경망의 능력과 RL의 의사결정 프레임워크를 통합하여 초인적 성능의 에이전트를 구축하는 방법을 제시하며, 특히 **일반화(Generalization)와 실험적 재현성**의 중요성을 강조한다. 이 연구는 향후 자율주행, 로보틱스, 정밀 의료 등 실세계의 복잡한 제어 문제를 해결하기 위한 학술적/기술적 토대를 제공한다.

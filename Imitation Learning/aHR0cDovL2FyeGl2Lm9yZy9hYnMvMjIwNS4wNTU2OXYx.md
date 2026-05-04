# Delayed Reinforcement Learning by Imitation

Pierre Liotet, Davide Maran, Lorenzo Bisi, Marcello Restelli (2022)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL) 환경에서 발생하는 **지연(Delay)** 문제, 특히 상태 관측 지연(State Observation Delay)과 액션 실행 지연(Action Execution Delay)을 해결하고자 한다. 일반적인 RL은 에이전트의 액션 결과가 즉각적으로 환경에 반영되고 관측된다는 가정을 전제로 하지만, 실제 로봇 제어, 트레이딩, 동적 시스템 등에서는 필연적으로 지연이 발생한다. 이러한 지연은 마르코프 가정(Markov assumption)을 위배하게 만들어, 기존의 RL 알고리즘들이 성능 저하를 겪거나 시스템의 불안정성을 초래하는 원인이 된다.

본 연구의 목표는 지연이 없는 환경(Undelayed Environment)에서 효율적인 정책(Policy)을 이미 알고 있거나 쉽게 학습할 수 있다는 가정하에, 이를 활용하여 지연이 존재하는 환경에서도 최적의 행동을 수행할 수 있는 정책을 효율적으로 학습하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 지연 환경에서의 RL 문제를 **모방 학습(Imitation Learning)** 문제로 전환하는 것이다. 저자들은 지연이 없는 환경의 전문가 정책(Expert Policy)을 모방함으로써 지연 환경에 적응하는 **DIDA(Delayed Imitation with Dataset Aggregation)** 알고리즘을 제안한다.

DIDA의 중심적인 직관은 다음과 같다. 지연 환경에서 직접 보상을 통해 학습하는 대신, 지연이 없는 환경에서 학습된 전문가가 현재의 '추정 상태'에서 어떻게 행동했을지를 모방하도록 학습시키는 것이다. 이를 위해 데이터셋 집계(Dataset Aggregation) 기법인 DAGGER를 활용하여, 학습자가 겪는 상태 분포의 변화(Distribution Shift) 문제를 해결하고 샘플 효율성을 극대화한다.

## 📎 Related Works

기존의 지연 RL 접근 방식은 크게 세 가지 방향으로 나뉜다.

1. **Memoryless 접근 방식**: 가장 최근에 관측된 상태에만 의존하여 정책을 결정한다. $\text{dSARSA}$와 같은 방식이 이에 해당하며, 업데이트 과정에서 지연을 고려하지만 근본적인 정보 손실이 존재한다.
2. **Augmented 접근 방식**: 최근 관측 상태와 그 이후에 취한 액션들의 시퀀스를 합쳐 **Augmented State**를 구성함으로써 문제를 다시 MDP로 변환한다. 하지만 지연 시간 $\Delta$가 커질수록 상태 공간이 기하급수적으로 증가하는 '차원의 저주(Curse of Dimensionality)' 문제가 발생한다.
3. **Model-based 접근 방식**: 환경 모델을 학습하여 현재의 알 수 없는 상태를 예측하고 이를 기반으로 액션을 선택한다. $\text{D-TRPO}$나 $\text{L2-TRPO}$가 대표적이며, 모델 설계의 복잡성과 계산 비용이 높다는 단점이 있다.

DIDA는 Augmented 방식의 차원의 저주와 Model-based 방식의 높은 계산 비용 및 복잡한 모델 설계 문제를 회피하면서도, 전문가 정책의 지식을 직접적으로 전이함으로써 효율적인 학습을 가능케 한다.

## 🛠️ Methodology

### 1. 시스템 구조 및 Augmented State

지연이 존재하는 DMDP(Delayed MDP)에서 에이전트는 현재 상태 $s$에 직접 접근할 수 없다. 대신 $\Delta$ 단계 전의 상태 $s_1$과 그 이후에 취한 액션 시퀀스 $(a_1, \dots, a_\Delta)$를 알 수 있다. 이를 통해 에이전트는 다음과 같은 **Augmented State** $x$를 구성한다.
$$x = (s_1, a_1, \dots, a_\Delta) \in S \times A^\Delta$$

또한, 현재 상태 $s$에 대한 확률 분포인 **Belief** $b(s|x)$를 정의하여, Augmented State $x$가 주어졌을 때 실제 현재 상태가 $s$일 확률을 계산한다.

### 2. DIDA 알고리즘 및 학습 절차

DIDA는 DAGGER 알고리즘을 기반으로 하며, 학습 과정은 다음과 같이 진행된다.

1. **전문가 학습**: 지연이 없는 환경에서 $\text{SAC}$ 등을 이용하여 전문가 정책 $\pi^E$를 먼저 학습한다.
2. **데이터 수집 및 반복 학습**:
    * $\beta_i$ 가중치를 사용하여 전문가 정책 $\pi^E$와 학습 중인 지연 정책 $\pi^I$의 혼합 정책으로 궤적을 샘플링한다.
    * 샘플링된 각 단계에서, 현재 상태 $s$에 대해 전문가가 내렸을 액션 $a^E \sim \pi^E(\cdot|s)$를 쿼리하여 데이터셋 $D$에 $(x, a^E)$ 쌍으로 저장한다. 이때 $x$는 해당 시점의 Augmented State이다.
    * 수집된 데이터셋 $D$를 사용하여 지연 정책 $\pi^I$를 지도 학습(Supervised Learning) 방식으로 업데이트한다.

### 3. 학습되는 정책의 본질

DIDA를 통해 학습되는 정책 $\pi^b$는 수학적으로 다음과 같은 형태를 띠게 된다.
$$\pi^b(a|x) = \int_S b(s|x) \pi^E(a|s) ds$$
즉, 현재 상태에 대한 믿음(Belief)을 바탕으로 전문가 정책의 기대값을 취하는 형태가 된다.

### 4. 비정수 지연(Non-integer Delays)으로의 확장

지연 $\Delta$가 정수가 아닌 경우, 저자들은 이를 두 개의 인터리브된(interleaved) MDP로 해석한다. 시간 인덱스 $t$와 $t+\Delta$를 가진 두 MDP가 동일한 전이 및 보상 함수를 공유한다고 가정하며, 지연 정책은 $M$에서 상태를 관측하고 $M_\Delta$에서 액션을 실행하는 형태로 구현된다.

## 📊 Results

### 1. 실험 설정

* **데이터셋 및 환경**: Pendulum, Mujoco(HalfCheetah, Walker2d, Reacher, Swimmer), Trading(EUR-USD 환율 데이터)
* **비교 대상(Baselines)**: $\text{M-TRPO}$, $\text{A-TRPO}$, $\text{D-TRPO}$, $\text{L2-TRPO}$, $\text{SARSA}$, $\text{dSARSA}$, $\text{M-SAC}$, $\text{A-SAC}$
* **측정 지표**: 평균 누적 보상(Mean Return) 및 샘플 효율성(학습 단계 수)

### 2. 주요 결과

* **수렴 속도**: Pendulum과 Mujoco 환경에서 DIDA는 다른 모든 베이스라인보다 훨씬 빠르게 수렴하며, 특히 Mujoco에서는 50만 단계 미만에서 최종 성능에 도달하는 뛰어난 샘플 효율성을 보였다.
* **지연에 대한 강건성**: 지연 시간 $\Delta$가 증가함에 따라 다른 알고리즘들은 성능이 급격히 하락하는 반면, DIDA는 가장 완만한 성능 저하를 보이며 강건함을 입증하였다.
* **실제 적용 가능성**: 트레이딩 작업에서 10초의 비정수 지연이 존재하는 상황에서도 긍정적인 수익률을 유지하였으며, 일부 구간에서는 지연이 없는 전문가보다 더 높은 성능을 보이기도 했다. 이는 모방 학습 과정이 일종의 정규화(Regularization) 역할을 하여 오버피팅을 방지했기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 1. 이론적 분석 및 성능 경계

본 논문은 $\text{Lipschitz MDP}$ 가정을 통해 지연 정책 $\pi^b$와 전문가 정책 $\pi^E$ 사이의 성능 차이를 분석하였다.

* **Corollary 6.1**에 따르면, 성능 손실은 지연 시간 $\Delta$에 선형적으로 비례한다.
* **Corollary 6.2**에서는 성능 손실이 현재 상태에 대한 Belief의 분산($\text{Var}[s|x]$)에 의존함을 보였다.

### 2. 전문가 정책의 매끄러움(Smoothness)과 Trade-off

이론적 분석 결과, 전문가 정책 $\pi^E$가 매끄러울수록(즉, Lipschitz 상수가 작을수록) 지연 환경에서의 성능 하한선이 높아진다. 이는 최적의 정책(Optimal Policy)이더라도 너무 급격하게 변하는 특성을 가졌다면, 지연 환경에서는 오히려 약간 sub-optimal 하더라도 매끄러운 정책을 모방하는 것이 더 유리할 수 있다는 중요한 통찰을 제공한다.

### 3. 한계점 및 논의

DIDA는 전문가 정책의 존재를 전제로 하므로, 전문가를 구하기 어려운 환경에서는 적용이 불가능하다. 또한, 학습되는 정책이 $\pi^b$ 형태(Belief의 기대값)를 띠므로, 일부 복잡한 MDP에서는 이론적으로 최적이 아닐 수 있다.

## 📌 TL;DR

본 논문은 지연이 존재하는 RL 문제를 해결하기 위해, 지연이 없는 환경의 전문가를 모방하는 **DIDA(Delayed Imitation with Dataset Aggregation)** 알고리즘을 제안한다. DAGGER 프레임워크를 통해 Augmented State에서 전문가의 액션을 모방함으로써 차원의 저주와 모델 설계의 복잡성을 동시에 해결하였다. 실험적으로는 로봇 제어 및 트레이딩 작업에서 기존 SOTA 방식보다 압도적인 샘플 효율성과 지연에 대한 강건성을 입증하였으며, 이론적으로는 전문가 정책의 Smoothness가 지연 환경의 성능에 미치는 영향을 분석하였다. 이 연구는 시뮬레이터를 통해 전문가를 쉽게 만들 수 있는 환경에서 지연 문제를 해결하는 매우 실용적인 방법론을 제시한다.

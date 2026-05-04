# Provably and Practically Efficient Adversarial Imitation Learning with General Function Approximation

Tian Xu, Zhilong Zhang, Ruishuo Chen, Yihao Sun, and Yang Yu (2024)

## 🧩 Problem to Solve

본 논문은 Adversarial Imitation Learning (AIL) 분야에서 이론적 분석과 실제 구현 사이의 거대한 간극을 해결하고자 한다. AIL은 신경망 기반의 함수 근사(Function Approximation)를 통해 실무적으로 큰 성공을 거두었으나, 이에 대한 이론적 연구는 주로 Tabular 설정이나 Linear function approximation과 같은 단순화된 시나리오에 국한되어 있었다. 

기존의 이론적 접근 방식들은 Count-based bonus나 Covariance-matrix-based bonus와 같이 특정 설정에 최적화된 복잡한 알고리즘 설계를 포함하고 있어, 이를 실제 신경망 기반의 시스템에 적용하기에는 상당한 어려움이 존재한다. 따라서 본 연구의 목표는 일반적인 함수 근사(General Function Approximation) 환경에서도 이론적으로 효율성이 증명 가능하며, 동시에 신경망을 통해 실무적으로 구현 가능한 AIL 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 이론적 보장과 실용적 효율성을 동시에 갖춘 **OPT-AIL (Optimization-based Adversarial Imitation Learning)** 알고리즘의 제안이다. 

OPT-AIL의 중심 아이디어는 보상 함수(Reward function)의 회복을 위해 온라인 최적화(Online Optimization)를 수행하고, Q-가치 함수(Q-value function)의 학습을 위해 낙관주의 정규화(Optimism-regularization)가 적용된 Bellman error 최소화를 수행하는 것이다. 이를 통해 일반적인 함수 근사 설정에서 처음으로 다항 시간 내의 Expert sample complexity와 Interaction complexity를 달성함을 이론적으로 증명하였다. 또한, 복잡한 보너스 설계 없이 두 가지 목적 함수(Objective)의 근사적 최적화만으로 구현이 가능하게 설계하여 deep AIL 방법론으로서의 실용성을 확보하였다.

## 📎 Related Works

기존의 AIL 연구는 크게 이론적 토대 마련과 실무적 알고리즘 개발이라는 두 갈래로 나뉜다. 이론적 연구들은 주로 transition function이 알려진 이상적인 설정이나 Tabular, Linear MDP 설정에서 분석되었으며, 최근에는 unknown transitions 환경에서의 온라인 AIL로 확장되었다. 그러나 이러한 연구들은 신경망과 같은 일반적인 함수 근사 설정에서의 효율성을 충분히 다루지 못했다.

실무적인 측면에서는 GAIL (Generative Adversarial Imitation Learning)이나 IQLearn (Inverse Q-Learning)과 같은 방법론들이 일반적인 함수 근사를 활용하여 뛰어난 성능을 보였으나, 이들은 일반적인 함수 근사 설정에서의 엄격한 이론적 보장이 결여되어 있었다. 본 논문은 이러한 이론과 실제의 괴리를 메우기 위해, RL에서의 일반 함수 근사 이론(예: Eluder dimension)을 AIL 설정으로 확장하여 적용함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조
OPT-AIL은 보상 함수와 정책을 반복적으로 업데이트하는 파이프라인을 가진다. 매 반복 단계 $k$마다 학습자는 환경과 상호작용하여 궤적(Trajectory)을 수집하고, 이를 바탕으로 보상 함수 $r^k$를 업데이트한 뒤, 해당 보상 하에서 최적의 정책 $\pi^k$를 도출한다.

### 1. 보상 함수의 업데이트 (Reward Update)
보상 함수 학습의 목표는 Expert와 학습 정책 간의 가치 차이를 최대화하는 보상을 찾는 것이다. 본 논문은 이를 온라인 최적화 문제로 정의하고 No-regret 알고리즘(예: FTRL)을 사용하여 해결한다. 
최적화하고자 하는 손실 함수 $L_i(r)$은 다음과 같이 정의된다:
$$L_i(r) = \hat{V}^{\pi_i}_r - \hat{V}^{\pi_E}_r$$
여기서 $\hat{V}^{\pi_i}_r$는 학습 정책 $\pi_i$의 추정 가치이며, $\hat{V}^{\pi_E}_r$는 전문가 데이터셋 $D_E$를 통한 전문가 정책의 추정 가치이다. 실무적으로는 FTRL을 적용하여 과거의 모든 손실 함수 합과 정규화 항 $\beta\psi(r)$을 최소화하는 방향으로 $r^k$를 업데이트한다. 이때 $\psi(r)$은 학습 안정성을 위해 Gradient Penalty를 사용한다.

### 2. 정책의 업데이트 (Policy Update)
추론된 보상 $r^k$ 하에서 정책 $\pi^k$를 학습하는 과정은 RL 문제로 귀결된다. OPT-AIL은 Bellman error를 최소화하고 동시에 탐색을 장려하기 위해 낙관적인 Q-가치 함수를 찾는 최적화 문제를 푼다.
최적화 목적 함수 $\mathcal{L}_k(Q)$는 다음과 같다:
$$\min_{Q \in \mathcal{Q}} \mathcal{L}_k(Q) := BE_k(Q) - \lambda \max_{a \in A} Q^1(s_1, a)$$
여기서 $BE_k(Q)$는 데이터셋 $D_k$와 보상 $r^k$에 대한 추정 제곱 Bellman error이며, $\lambda$는 정규화 계수이다. $\max Q^1$ 항은 Q-가치를 낙관적으로 추정하게 하여 에이전트가 더 효율적으로 탐색하도록 유도한다. 최종 정책 $\pi^k$는 학습된 $Q^k$에 대한 Greedy 정책으로 결정된다.

### 3. 이론적 보장 및 복잡도
본 논문은 보상 클래스 $\mathcal{R}$의 realizability, Q-가치 클래스 $\mathcal{Q}$의 realizability 및 Bellman completeness, 그리고 MDP의 Low generalized eluder coefficient ($d_{GEC}$) 가정을 전제로 이론적 분석을 수행하였다. 
분석 결과, OPT-AIL은 다음과 같은 복잡도를 가진다:
- **Expert Sample Complexity:** $\tilde{O}\left(\frac{H^2 \log(\max_{h} N(R^h))}{\epsilon^2}\right)$
- **Interaction Complexity:** $\tilde{O}\left(\frac{H^4 d_{GEC} \log(\max_{h} N(Q^h)N(R^h)) + H^2}{\epsilon^2}\right)$
이는 일반 함수 근사 설정에서 AIL의 효율성을 증명한 첫 번째 결과이다.

## 📊 Results

### 실험 설정
- **데이터셋:** feature-based DMControl 벤치마크의 8가지 연속 제어 작업(Continuous Control Tasks)을 사용하였다.
- **기준선(Baselines):** BC, IQLearn, PPIL, FILTER, HyPE 등 최신 Deep AIL 및 IL 방법론들과 비교하였다.
- **지표:** 전문가 궤적 수에 따른 Return 값과 환경 상호작용 횟수에 따른 학습 곡선을 측정하였다.

### 정량적 및 정성적 결과
1. **Expert Sample Efficiency:** 전문가 데이터가 매우 제한적인 상황(예: 궤적이 1개인 경우)에서도 OPT-AIL은 Finger Spin, Walker Run, Hopper Hop 등의 작업에서 전문가 수준 또는 그에 근접한 성능을 보였으며, 이는 기존 SOTA 방법론들을 상회하는 결과이다.
2. **Interaction Efficiency:** 동일한 전문가 데이터 조건에서 OPT-AIL은 다른 Deep AIL 방법들보다 더 적은 환경 상호작용만으로도 높은 성능에 도달하였다. 특히 Hopper Hop과 Walker Run에서 상호작용 효율성이 매우 높게 나타났다.
3. **BC와의 비교:** BC(Behavioral Cloning) 대비 월등한 성능을 보였는데, 이는 이론적으로 분석된 BC의 Compounding error 문제를 OPT-AIL이 효과적으로 완화하고 있음을 입증한다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 논문은 AIL의 이론적 분석 대상이었던 단순 모델(Tabular, Linear)을 넘어, 실제 딥러닝에서 사용하는 General Function Approximation 설정에서도 다항 시간 복잡도의 효율성을 증명하였다는 점에서 학술적 가치가 매우 크다. 특히, 이론적 증명에 그치지 않고 이를 신경망으로 구현 가능한 형태로 단순화하여 실무적인 성능 향상까지 이끌어낸 점이 인상적이다.

### 한계 및 비판적 해석
이론적 증명을 위해 도입된 **Bellman completeness (Assumption 3)** 가정이 주요 한계로 지적된다. 이는 Q-가치 함수 클래스가 Bellman 연산에 대해 닫혀 있어야 함을 의미하는데, 실제 신경망 구조가 이 조건을 완벽히 만족하는지는 불분명하다. 저자들 또한 이를 인정하며 향후 연구에서 이 가정을 제거하는 방향으로 확장할 필요가 있음을 언급하였다. 또한, Tabular 설정에서 달성된 더 낮은 복잡도($O(H^{3/2}/\epsilon)$)를 일반 함수 근사 설정에서도 달성할 수 있을지는 여전히 미해결 과제로 남아 있다.

## 📌 TL;DR

본 논문은 일반적인 함수 근사 환경에서 이론적으로 효율성이 증명되고 실무적으로 구현 가능한 AIL 알고리즘인 **OPT-AIL**을 제안한다. 온라인 최적화 기반의 보상 학습과 낙관적 Bellman error 최소화 기반의 정책 학습을 결합하여, 전문가 데이터와 환경 상호작용 모두에서 높은 효율성을 달성하였다. 이는 AIL의 이론과 실제 사이의 간극을 좁힌 중요한 연구이며, 향후 데이터 효율적인 로보틱스 제어나 자율 주행 시스템의 모방 학습 연구에 핵심적인 기초가 될 가능성이 높다.
# Learning the Value Systems of Agents with Preference-based and Inverse Reinforcement Learning

Andrés Holgado-Sánchez, Holger Billhardt, Alberto Fernández, Sascha Ossowski (2026)

## 🧩 Problem to Solve

본 논문은 자율 소프트웨어 에이전트가 인간의 윤리적 원칙과 도덕적 가치에 부합하도록 설계하는 **Value Alignment**(가치 정렬) 문제를 다룬다. 특히, 서로 다른 사용자가 서로 다른 가치 체계(Value Systems)를 가질 수 있고, 특정 맥락에서 가치의 정확한 의미를 계산 가능한 방식으로 정의하기 어렵다는 점이 핵심 난제이다.

기존의 가치 추정 방식은 설문 조사와 같은 인간의 수동 설계에 의존하여 확장성이 낮으며, 단순히 행동을 모방하는 모방 학습(Imitation Learning)은 학습된 행동의 근거가 되는 윤리적 원칙을 명시적으로 설명할 수 없다는 한계가 있다. 따라서 본 연구의 목표는 인간의 시연(Demonstrations)과 관찰을 통해 **가치 접지 함수(Value Grounding Function)**와 **가치 체계(Value System)**를 자동으로 학습하는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 가치 학습 문제를 두 단계의 독립적인 과정으로 분리하여 해결하는 것이다.

1. **Value Grounding Learning**: 특정 도메인에서 각 가치가 개별적으로 어떻게 정의되는지(즉, 상태-행동 쌍이 특정 가치와 얼마나 정렬되는지)를 학습하여 공통의 가치 접지 함수를 구축한다.
2. **Value System Identification**: 학습된 접지 함수를 바탕으로, 개별 에이전트가 여러 가치 중 무엇을 더 중요하게 여기는지에 대한 가중치(Weight)를 추론하여 에이전트 고유의 가치 체계를 식별한다.

이를 위해 본 연구는 순차적 의사결정 문제를 **Multi-objective Markov Decision Processes (MOMDP)**로 정식화하고, **Preference-based RL (PbRL)**과 **Inverse RL (IRL)**을 결합한 프레임워크를 제안한다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 바탕으로 차별점을 제시한다.

* **Agreement Technologies (AT)**: 에이전트 간의 합의를 도출하는 기술로, 본 연구는 AT의 상위 레이어에서 책임 있는 자율성(Responsible Autonomy)을 확보하기 위해 가치 인식(Value-awareness) 기능을 통합하고자 한다.
* **Value Awareness & Ethical Decision-making**: 가치를 명시적 표현으로 모델링하려는 시도들이 있었으나, 대부분 수동으로 설계되어 **Reward Misspecification**(보상 오설정) 위험이 컸다.
* **Inverse Reinforcement Learning (IRL)**: 관찰된 행동에서 보상 함수를 추출하는 기법이다. 하지만 단일 보상 함수만으로는 행동의 근거가 되는 개별 윤리적 가치를 구분해낼 수 없다.
* **Preference-based RL (PbRL)**: 궤적 간의 쌍별 비교(Pairwise Comparison)를 통해 보상을 학습하는 방식으로, 본 연구는 이를 가치 접지 함수를 학습하는 데 활용하여 정량적인 가치 정렬 정도를 추정한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조: MVDP (Markov Value Decision Process)

본 논문은 환경을 **Markov Value Decision Process (MVDP)**로 정의한다. 이는 다음과 같은 튜플로 구성된 MOMDP의 일종이다: $(S, A, T, V, \mathbf{R}^V)$.

* $S, A, T$: 각각 상태 집합, 행동 집합, 전이 함수이다.
* $V = \{v_1, \dots, v_m\}$: 도메인에서 고려되는 $m$개의 가치 레이블 집합이다.
* $\mathbf{R}^V = (R^{v_1}, \dots, R^{v_m})$: 각 가치에 대응하는 보상 함수의 벡터로, 이것이 곧 **Grounding Function** 역할을 수행한다.

### 2. 가치 접지 학습 (Value Grounding Learning)

각 가치 $v_i$에 대한 보상 함수 $R^{v_i}$를 학습하기 위해 PbRL 기반의 접근법을 사용한다.

* **입력 데이터**: 궤적 쌍 $(\tau, \tau')$과 그들 사이의 정량적 선호도 $y \in [0, 1]$가 포함된 데이터셋 $D^{v_i}$를 사용한다.
* **Bradley-Terry 모델**: 궤적 $\tau$가 $\tau'$보다 더 선호될 확률을 다음과 같이 정의한다.
    $$p(\tau > \tau' | \hat{R}^{v_i}) = \frac{\exp \hat{R}^{v_i}(\tau)}{\exp \hat{R}^{v_i}(\tau) + \exp \hat{R}^{v_i}(\tau')}$$
* **손실 함수**: 교차 엔트로피(Cross-Entropy) 손실 함수를 최소화하여 신경망 파라미터 $\hat{\theta}$를 학습한다.
    $$L(\hat{\theta}) = -\frac{1}{|D^{v_i}|} \sum_{(\tau, \tau', y) \in D^{v_i}} y \log p(\tau > \tau' | \hat{R}^{\hat{\theta}_{v_i}}) + (1-y) \log(1 - p(\tau > \tau' | \hat{R}^{\hat{\theta}_{v_i}}))$$

### 3. 가치 체계 식별 (Value System Identification)

학습된 $\mathbf{R}^V$를 바탕으로 에이전트 $j$의 개별 가치 체계(가중치 $W_j$)를 추론한다.

* **선형 스칼라화 (Linear Scalarization)**: 에이전트 $j$의 통합 보상 함수 $R_j$는 각 가치 보상의 가중 합으로 표현된다.
    $$R_j(s, a) = W_j \cdot \mathbf{R}^V(s, a) = \sum_{i=1}^m w_{j,i} R^{v_i}(s, a)$$
* **Deep Maximum Entropy IRL**: 관찰된 에이전트의 궤적 $\mathcal{T}_j$를 통해 실제 정책 $\pi_j$와 학습된 정책 $\hat{\pi}_j$ 사이의 **State-Action Visitation Counts** 차이를 최소화하는 $W_j$를 찾는다.
* **최적화 목표**: 방문 빈도 행렬의 차이인 TVC(Total Visitation Count)를 줄이는 방향으로 가중치를 업데이트한다.

## 📊 Results

### 1. 실험 설정

두 가지 시뮬레이션 시나리오에서 검증을 수행하였다.

* **Firefighters**: 소방관이 인명 구조(Proximity)와 전문적 절차(Professionalism) 사이에서 의사결정을 내리는 환경이다. (가치 2개)
* **Roadworld**: 상하이 도로 네트워크에서 지속가능성(Sustainability), 쾌적함(Comfort), 효율성(Efficiency)을 고려하여 경로를 선택하는 환경이다. (가치 3개)

### 2. 주요 결과

* **접지 함수 학습 정확도**: Roadworld에서는 거의 완벽한 보상 함수 복원을 보였으며, Firefighters에서도 98% 이상의 선호도 예측 정확도를 달성하였다. 이는 PbRL 기반의 접지 학습이 효과적임을 입증한다.
* **가치 체계 식별 성능**:
  * **Firefighters**: 학습된 가중치 $W_j$가 원래 에이전트의 가중치와 거의 일치하였으며, TVC 에러가 0.005 미만으로 매우 낮게 나타났다.
  * **Roadworld**: 정책 수준에서는 완벽하게 복원되었으나, 일부 가중치 값이 원래 값과 다르게 추론되는 경우가 발생했다. 이는 '지속가능성'과 '효율성' 같은 가치들이 서로 강한 양의 상관관계(Positive Correlation)를 가지고 있어, 서로 다른 가중치 조합으로도 동일한 행동이 나올 수 있기 때문이다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 연구는 가치 학습을 '의미 정의(Grounding)'와 '우선순위 결정(System Identification)'으로 분리함으로써, AI 모델의 **해석 가능성(Interpretability)**을 크게 높였다. 단순히 행동을 모방하는 것이 아니라, "이 에이전트는 효율성보다 지속가능성을 2배 더 중요하게 생각한다"와 같은 명시적인 가중치 형태로 가치 체계를 추출할 수 있기 때문이다.

### 한계 및 비판적 해석

* **동질적 사회 가정**: 모든 에이전트가 가치의 의미(Grounding)에 대해서는 합의하고 있으며, 가중치만 다르다는 가정을 전제로 한다. 하지만 다문화 사회와 같이 가치 자체의 정의가 서로 다른 경우 본 프레임워크를 그대로 적용하기 어렵다.
* **선형 결합의 단순성**: 가치 체계를 단순한 선형 가중 합으로 모델링하였다. 실제 인간의 가치 판단은 상황에 따라 가중치가 변하거나 비선형적인 상호작용을 일으킬 가능성이 높으나, 이를 반영하지 못했다.
* **데이터 의존성**: PbRL 단계에서 많은 양의 쌍별 비교 데이터가 필요하며, 실제 환경에서 인간 전문가로부터 이러한 데이터를 수집하는 것은 비용이 많이 드는 작업이다.

## 📌 TL;DR

본 논문은 인간의 행동 관찰을 통해 AI 에이전트의 도덕적 가치 체계를 학습하는 프레임워크를 제안한다. **PbRL을 통해 개별 가치의 의미(Grounding)를 먼저 학습**하고, **Deep MaxEnt IRL을 통해 에이전트별 가치 가중치(Value System)를 추론**하는 2단계 구조를 가진다. 실험을 통해 제안 방법이 에이전트의 가치 우선순위를 정확하게 식별하고 행동을 복원할 수 있음을 보였으며, 이는 가치 정렬된(Value-aligned) AI 시스템 구축을 위한 해석 가능한 방법론을 제공한다.

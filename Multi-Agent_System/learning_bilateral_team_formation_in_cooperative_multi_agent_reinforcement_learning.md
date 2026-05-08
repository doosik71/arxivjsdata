# Learning Bilateral Team Formation in Cooperative Multi-Agent Reinforcement Learning

Koorosh Moslemi, Chi-Guhn Lee (2025)

## 🧩 Problem to Solve

본 논문은 협력적 다중 에이전트 강화학습(Cooperative Multi-Agent Reinforcement Learning, MARL) 환경에서 에이전트들이 어떻게 효율적으로 팀을 구성(Team Formation)할 것인가에 대한 문제를 다룬다.

기존의 팀 구성 연구들은 주로 한쪽 방향으로만 그룹을 묶는 Unilateral grouping, 사전에 정의된 팀(Predefined teams), 또는 에이전트의 수가 고정된 설정(Fixed-population settings)에 집중되어 있었다. 하지만 실제 동적인 환경에서는 에이전트들의 집합이 유동적이며, 팀 구성의 결정이 양방향(Bilateral)으로 이루어질 때 어떤 알고리즘적 특성이 정책 성능과 일반화(Generalization) 능력에 영향을 미치는지에 대한 연구가 부족한 실정이다.

따라서 본 논문의 목표는 동적인 다중 에이전트 시스템에서 양방향 팀 구성을 학습하는 프레임워크를 제안하고, 특히 매칭 알고리즘의 안정성(Stability)이 정책 성능 및 학습되지 않은 에이전트 구성(Unseen agent compositions)에 대한 일반화 성능에 미치는 영향을 분석하는 것이다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 다음과 같다.

1. **양방향 매칭 기반의 팀 구성 연구**: MARL의 팀 구성 문제를 서로 겹치지 않는 두 에이전트 집합 간의 Bilateral matching 문제로 정의하였다. 특히, 안정적 매칭(Stable matching) 알고리즘을 사용했을 때가 불안정적 매칭을 사용했을 때보다 새로운 에이전트 구성에 대해 더 뛰어난 일반화 성능을 보임을 실험적으로 증명하였다.
2. **어텐션 기반 가치 분해(Attention-based Value Decomposition) 프레임워크 제안**: 에이전트 간의 어텐션 점수를 선호도(Preference)로 해석하고, 이를 매칭 알고리즘에 활용하는 구조를 제안하였다. 동적인 에이전트 인구 수를 처리하기 위해 가치 네트워크(Value Network)와 믹싱 네트워크(Mixing Network)에 Group-aware한 구조를 도입하였다.

## 📎 Related Works

### 기존 연구 및 한계

* **팀 구성 방법론**: GoMARL은 고정된 인구 집합 내에서 휴리스틱 기반의 적응형 그룹 구조를 학습하지만, 인구 수가 변하는 상황에 대응하기 어렵다. COPA는 코치 에이전트가 전략을 방송하는 방식을 사용하나, 훈련 시 팀이 미리 정의되어 있어야 한다는 제약이 있다. SOG는 한쪽 집합(Conductor)이 다른 쪽을 선택하는 Unilateral matching 방식을 사용한다.
* **매칭 시장(Matching Market)**: 경제학 분야에서 Gale & Shapley의 안정적 매칭 이론은 널리 연구되었으며, 최근에는 밴딧(Bandit) 프레임워크를 통해 선호도를 학습하는 연구들이 진행되었다. 그러나 이러한 안정성 개념을 MARL의 팀 구성과 결합하여 분석한 연구는 기존에 없었다.

### 차별점

본 논문은 단순히 그룹을 나누는 것을 넘어, 매칭의 '안정성'이라는 수학적 특성이 MARL의 비정상성(Non-stationarity) 문제와 결합했을 때 정책의 안정성과 일반화에 어떤 영향을 주는지를 집중적으로 탐구한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

본 프레임워크는 REFIL을 기반으로 하며, 에이전트를 리더(Leader, $L$)와 팔로워(Follower, $F$)의 두 집합으로 분리하여 이들 간의 매칭을 통해 팀을 형성한다. 전체 구조는 크게 에이전트 유틸리티 네트워크와 하이퍼네트워크의 수정으로 이루어진다.

### 주요 구성 요소 및 역할

1. **Agent Utility Network**:
    * **MHA (Multi-Head Attention)**: 에이전트 간의 관계를 파악하며, 여기서 도출된 어텐션 가중치를 팀 구성을 위한 선호도(Preference)로 사용한다.
    * **Encoder-Decoder**: 그룹-인지 인코더 $f_e$는 동일 그룹 내 에이전트들의 임베딩은 유사하게, 다른 그룹 간에는 다르게 생성하도록 학습된다. 디코더는 이 임베딩을 사용하여 유틸리티 네트워크의 마지막 층 파라미터를 생성한다.
2. **Group-aware Hypernetwork**:
    * 인코더 $f_e$에서 생성된 에이전트 임베딩들을 Max pooling하여 그룹별 상태를 생성하고, 이를 통해 믹싱 네트워크 $f_{mix}$의 가중치를 생성한다. 이는 가치 분해 과정에서 그룹 정보를 직접적으로 반영하게 한다.

### 매칭 알고리즘 (Matching Algorithms)

에이전트 간 선호도 행렬을 바탕으로 두 가지 매칭 방식을 비교한다.

* **Order Oriented Matching (OOM)**: Gale & Shapley의 Deferred Acceptance(DA) 메커니즘을 사용한다. 리더가 선호하는 팔로워에게 제안하고, 팔로워는 더 좋은 제안이 오면 기존 매칭을 취소하고 새 제안을 수락하는 방식이다. 결과적으로 어떤 에이전트도 현재 매칭보다 더 선호하는 상대와 서로 매칭되기를 원하는 상태가 없는 **안정적 매칭(Stable Matching)**을 보장한다.
* **Score Oriented Matching (SOM)**: 어텐션 점수의 절대적인 합(Mutual score)을 기준으로 매칭한다. 팔로워가 자신과 가장 높은 상호 점수를 가진 리더를 선택하는 단순한 방식이며, 이는 **불안정적 매칭(Unstable Matching)**으로 이어진다.

### 학습 목표 및 손실 함수

본 모델은 표준 Q-learning 손실 함수 $L_Q$ 외에 두 가지 추가 손실 함수를 사용한다.

1. **Auxiliary Loss ($L_{aux}$)**: 가치 함수를 $2|A|$개의 인자로 분해하여 학습함으로써 그룹 구조를 강화한다.
    * $L_{TD} = (1-\lambda)L_Q + \lambda L_{aux}$
2. **Similarity-Diversity Loss ($L_{SD}$)**: 인코더 $f_e$가 그룹 내 유사성과 그룹 간 다양성을 확보하도록 강제한다.
    $$L_{SD}(\theta_e) = \mathbb{E}_B \left( \sum_{i \neq j} I(i,j) \cdot \text{cosine}(f_e(h_i; \theta_e), f_e(h_j; \theta_e)) \right)$$
    여기서 $I(i,j)$는 에이전트 $i, j$가 같은 그룹이면 $-1$, 다른 그룹이면 $1$의 값을 가진다.

최종 손실 함수는 다음과 같다.
$$L = L_{TD} + L_{SD}$$

## 📊 Results

### 실험 설정

* **데이터셋 및 환경**: StarCraft Multi-Agent Challenge (SMAC)를 사용하였으며, SZ, CSZ, MMM 시나리오에서 평가하였다.
* **설정**: 훈련 시에는 3~5명의 에이전트를 사용하고, 평가 시에는 6~8명의 에이전트를 배치하여 일반화 성능을 측정하였다.
* **비교 대상**: MIPI, REFIL, AQMIX, CollaQ, MAPPO 등 기존 MARL 베이스라인과 제안 방법의 OOM, SOM을 비교하였다.

### 주요 결과

* **정량적 성능**: 대부분의 평가 시나리오(9개 중 6개)에서 제안 방법의 최적 구성이 베이스라인보다 높은 승률을 기록하였다.
* **OOM vs SOM**: 특히 27개의 평가 구성 중 26개에서 **OOM이 SOM보다 일관되게 높은 성능**을 보였다. 훈련 단계에서의 성능 차이는 크지 않았으나, 에이전트 수가 늘어난 평가 단계(Generalization)에서 OOM의 우위가 뚜렷하게 나타났다.

## 🧠 Insights & Discussion

### 분석 및 해석

본 논문은 OOM의 우수한 일반화 능력이 매칭의 **안정성(Stability)**에서 기인한다고 분석한다. 안정적 매칭은 에이전트가 현재의 파트너를 버리고 다른 파트너로 갈아타려는 유인(Incentive)을 제거한다. 이는 MARL의 동적인 환경에서 팀 구성이 빈번하게 바뀌어 발생하는 정책의 불안정성(Inefficient policy switching)을 억제함으로써, 더 견고한 정책 학습과 일반화를 가능하게 한다.

### 한계 및 향후 연구

* **Distracted Attention 문제**: 에이전트의 시야 범위가 넓어질 때 무관한 정보에 집중하게 되는 문제가 제기된 바 있다. OOM은 점수의 절대값이 아닌 '상대적 순서'만을 사용하므로, SOM보다 이러한 노이즈에 더 강건할 가능성이 있으며 이는 향후 연구 과제로 제시되었다.
* **리더 선정 및 규모**: 현재는 고정된 수의 리더를 설정하거나 단순하게 지정하고 있으나, 리더의 수 $|L|$을 원칙적으로 튜닝하거나 최적으로 선택하는 방법론에 대한 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 MARL에서 에이전트 팀 구성을 **양방향 매칭(Bilateral Matching)** 문제로 정의하고, 매칭의 **안정성(Stability)**이 성능과 일반화에 미치는 영향을 분석하였다. 실험 결과, Gale-Shapley 알고리즘 기반의 **안정적 매칭(OOM)**이 단순 점수 기반 매칭(SOM)보다 학습되지 않은 에이전트 구성에 대해 훨씬 뛰어난 일반화 성능을 보임을 입증하였다. 이 연구는 동적 인구 환경의 MARL 시스템에서 팀 구성 알고리즘의 선택이 정책의 안정성에 결정적인 역할을 할 수 있음을 시사한다.

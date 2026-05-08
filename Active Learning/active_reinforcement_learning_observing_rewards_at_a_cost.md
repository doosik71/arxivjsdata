# Active Reinforcement Learning: Observing Rewards at a Cost

David Krueger, Jan Leike, Owain Evans, John Salvatier (2020)

## 🧩 Problem to Solve

본 논문은 보상 신호를 관찰하는 데 비용이 발생하는 환경에서의 강화학습, 즉 Active Reinforcement Learning (ARL) 문제를 다룬다. 일반적인 강화학습에서는 에이전트가 행동을 취한 후 보상을 즉각적으로, 그리고 무료로 관찰한다고 가정한다. 그러나 실제 환경에서는 인간 전문가의 피드백을 받거나 정밀한 의료 검사를 수행하는 것과 같이 보상 정보를 얻는 행위 자체가 상당한 비용(Query Cost)을 초래하는 경우가 많다.

따라서 본 연구의 핵심 문제는 에이전트가 보상 정보를 획득하기 위해 비용 $c > 0$를 지불할 것인지, 아니면 지불하지 않고 현재의 불완전한 정보로 행동할 것인지를 결정하는 최적의 전략을 찾는 것이다. 논문의 목표는 전체 보상의 합에서 총 쿼리 비용을 뺀 값, 즉 $\text{Total Reward} - \text{Total Query Cost}$를 최대화하는 메커니즘을 제안하고 평가하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여는 보상 관찰 비용이 존재하는 상황에서 정보의 장기적 가치를 정량화하려는 시도와 이를 해결하기 위한 휴리스틱 알고리즘의 제안이다.

1. **Active Multi-Armed Bandits (MAB)에서의 접근**: 쿼리 비용이 존재하는 밴딧 문제의 특성을 분석하고, 정보 획득의 가치를 추정하여 쿼리 중단 시점을 결정하는 Mind-changing cost heuristic (MCCH) 알고리즘을 제안하였다.
2. **Tabular MDP에서의 접근**: 모델 기반 강화학습 프레임워크 내에서 쿼리 전략을 최적화하는 두 가지 방법론인 Simulated Query Rollout (SQR)과 Value of Information (VOI) 기반의 쿼리 횟수 결정 방법을 제안하였다.
3. **ARL의 이론적/실험적 분석**: ARL이 일반적인 RL이나 Active Learning과 어떻게 다른지 정의하고, 특히 MDP에서 전이 함수(Transition Function) 학습과 보상 함수 학습의 분리 가능성을 제시하였다.

## 📎 Related Works

논문에서는 인간의 목적을 보상 함수로 변환하는 어려움을 다룬 기존 연구들을 언급한다.

- **Inverse Reinforcement Learning (IRL)** 및 **Preference-based RL**: 보상 함수를 직접 설계하는 대신 전문가의 시연이나 선호도를 통해 학습하는 방식이다.
- **TAMER Framework**: 인간 교사가 온라인으로 보상을 제공하는 설정이나, 주로 보상 자체가 아닌 행동의 가치($Q^*$)에 대한 신호를 제공한다는 점에서 ARL과 차이가 있다.
- **Active Reward Learning**: 로봇 그리핑 작업 등에 적용된 사례가 있으며 Gaussian Process 등을 사용하지만, 본 논문이 다루는 이산적(Discrete) 문제에는 직접 적용하기 어렵다.

ARL의 차별점은 보상을 관찰하는 행위 자체에 명시적인 '비용'이 할당되어 있으며, 에이전트가 이 비용과 정보 획득을 통한 미래 수익 사이의 트레이드오프를 스스로 결정해야 한다는 점에 있다.

## 🛠️ Methodology

### 1. Active Multi-Armed Bandits

에이전트는 $K$개의 팔 중에서 하나를 선택하며, 보상을 관찰하려면 비용 $c$를 지불해야 한다. 목표는 기대 후회(Expected Regret)를 최소화하는 것이다:
$$\text{Regret}_n = E \left[ \sum_{t=1}^n (\mu^* - R_t + cQ_t) \right]$$
여기서 $Q_t$는 $t$ 시점에 보상을 관찰했을 때 1, 아니면 0이다.

**Mind-changing cost heuristic (MCCH):**
이 알고리즘은 현재 최선의 팔($\hat{i}$)을 유지하는 것과 정보를 더 수집하여 팔을 바꿀 가능성을 비교한다.

- $\hat{m}$: 두 번째로 좋은 팔의 사후 평균이 최선의 팔의 사후 평균으로 이동하기 위해 필요한 연속 쿼리 횟수의 추정치이다.
- $\widehat{\text{Regret}}_n(\hat{i})$: 현재 최선의 팔 $\hat{i}$에 고정했을 때의 기대 후회이다.
- **결정 규칙**: 다음 조건이 만족될 때만 쿼리 비용을 지불한다:
$$c\hat{m} < \alpha \widehat{\text{Regret}}_n(\hat{i})$$
여기서 $\alpha$는 쿼리 의욕을 조절하는 하이퍼파라미터이다.

### 2. Active RL in MDPs

Tabular MDP 환경에서 Posterior Sampling for RL (PSRL)을 기반으로 하며, 모델 기반으로 접근한다.

**Simulated Query Rollout (SQR):**
특정 쿼리 전략(예: 상태-행동쌍당 $N$번 쿼리)의 성능을 에이전트의 사후 분포에서 샘플링된 여러 환경에서 시뮬레이션하여 최적의 $N$을 선택한다. ASQR(Approximate SQR)은 에이전트가 상태를 실제로 방문하지 않고도 보상을 직접 학습한다고 가정하여 계산 효율을 높였다.

**Value of Information (VOI):**
각 상태-행동 $(s, a)$의 보상을 아는 것이 얼마나 가치 있는지를 독립적으로 계산한다.
$$\text{VOI} = E_{P(M)} [V^M(\pi_{\text{inf}}) - V^M(\pi_{\text{ign}})]$$

- $\pi_{\text{inf}}$: 보상 $R(s, a)$를 알고 있는 정보-충분(informed) 에이전트의 정책.
- $\pi_{\text{ign}}$: 보상 $R(s, a)$를 모르는 정보-부족(ignorant) 에이전트의 정책.

이 VOI 값을 이용하여 각 상태-행동별 최적 쿼리 횟수 $N(s, a)$를 다음과 같이 결정한다:
$$N(s, a) = \frac{k \cdot E \cdot \text{VOI}[R(s, a)]}{c}$$
여기서 $E$는 남은 에피소드 수, $c$는 쿼리 비용, $k$는 하이퍼파라미터이다.

## 📊 Results

### 1. Active Bandits 실험

- **설정**: Bernoulli 분포를 가진 팔, $\text{horizon} = 10^4$, 쿼리 비용 $c \in \{2, 50\}$.
- **결과**: MCCH는 Knowledge Gradient나 $1/t$-query 방식보다 절대적인 성능이 항상 최고는 아니었으나, 다양한 쿼리 비용 설정에서도 매우 강건(Robust)한 성능을 보였다. 특히 하이퍼파라미터 $\alpha$의 변화에 민감하지 않아 실용적임을 입증하였다.

### 2. MDP 실험

- **환경**: Chain (길이 10), Long-Y (길이 10, 가지 구조), 4x4 Gridworld.
- **결과**:
  - **Long-Y 환경**: VOI 알고리즘이 가장 우수한 성능을 보였다. 이는 VOI 방식이 어떤 상태가 '불가피하게 방문해야 하는 상태(unavoidable states)'인지 파악하여, 굳이 쿼리 비용을 들여 보상을 확인할 필요가 없는 구간을 효율적으로 제외했기 때문이다.
  - **Chain 환경**: ASQR 기반 알고리즘이 더 나은 성능을 보였다.
  - **전반적 관찰**: VOI 알고리즘은 리턴 값은 높았으나 쿼리 횟수가 많아 최종 수익(보상 - 비용) 면에서는 손해를 보는 경우가 있었다. 이는 $k$ 값을 조정하여 개선 가능할 것으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 ARL이 단순한 RL의 변형이 아니라, 정보 획득의 경제학을 다루는 문제임을 보여준다.

- **강점**: 전이 함수 $P$는 보상 비용 없이 학습할 수 있다는 점을 이용하여, 환경의 구조를 먼저 파악한 뒤 필요한 곳에만 쿼리를 집중하는 전략이 유효함을 보였다. 특히 '불가피한 상태'에 대한 통찰은 매우 중요하다.
- **한계**: 제안된 알고리즘들이 주로 모델 기반(Model-based)이며 Tabular 설정에 국한되어 있다. 또한, 보상 관찰에 지연(delay)이 발생하는 현실적인 제약 조건을 고려하지 않았다.
- **비판적 해석**: VOI 기반 방식이 이론적으로는 우수해 보이지만, 실제 구현 시 계산 복잡도가 높고 하이퍼파라미터 $k$에 따라 성능 편차가 크다. 또한, 초기 에피소드에서 쿼리를 너무 적게 수행하는 경향이 있는데, 이는 사후 분포의 불확실성을 고려한 Confidence-bound 기반의 접근법으로 보완될 필요가 있다.

## 📌 TL;DR

본 연구는 보상 신호를 얻기 위해 비용을 지불해야 하는 **Active Reinforcement Learning (ARL)** 프레임워크를 제안하였다. 밴딧 문제에서는 쿼리 중단 시점을 결정하는 **MCCH**를, MDP 문제에서는 정보 가치를 기반으로 쿼리 횟수를 조절하는 **SQR** 및 **VOI** 알고리즘을 제시하였다. 특히 환경의 구조(전이 함수)를 활용해 불필요한 쿼리를 줄이는 것이 효율적임을 입증하였으며, 이는 향후 고비용 피드백이 필요한 실제 RL 시스템(인간-AI 상호작용, 의료 최적화 등) 설계에 중요한 기초가 될 수 있다.

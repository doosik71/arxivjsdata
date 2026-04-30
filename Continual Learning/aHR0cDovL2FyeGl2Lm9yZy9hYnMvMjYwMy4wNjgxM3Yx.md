# REINFORCING THE WORLD’S EDGE: A CONTINUAL LEARNING PROBLEM IN THE MULTI-AGENT-WORLD BOUNDARY

Dane Malenfant (2026)

## 🧩 Problem to Solve

본 논문은 강화학습(RL)에서 에이전트가 학습한 결정 구조(decision structure)가 에피소드 간에 얼마나 유지될 수 있는지, 그리고 이것이 **에이전트-세계 경계(agent–world boundary)**를 어떻게 설정하느냐에 따라 어떻게 달라지는지를 분석한다. 

일반적인 단일 에이전트 환경에서는 세계의 역학(dynamics)이 고정되어 있어 성공적인 궤적들 사이의 공통된 구조를 재사용할 수 있다. 하지만 분산형 다중 에이전트 강화학습(Decentralized MARL) 환경에서 다른 에이전트(peer agent)를 '세계'의 일부로 간주할 경우, 상대 에이전트의 정책 업데이트가 곧 세계의 역학 변화로 이어진다. 이로 인해 에이전트-세계 경계가 불안정해지며, 이는 외부적인 작업 전환이 없더라도 내생적으로 **지속적 학습(Continual Learning, CL)** 문제로 전이되는 결과를 초래한다. 논문의 목표는 이러한 경계 표류(boundary drift) 현상을 이론적으로 정형화하고, 재사용 가능한 구조의 소실을 정량화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1.  **Invariant Core의 정의 및 증명**: 성공적인 궤적들이 공유하는 최대 부분 시퀀스(maximal subsequences)인 'Invariant Core'를 정의하고, 정적인 MDP 환경에서 이의 존재성을 증명하였다.
2.  **MARL의 CL 관점 재해석**: 분산형 MARL에서 상대 에이전트를 세계에 포함시킬 경우, 상대의 정책 변화가 유도된 MDP(induced MDP)의 시퀀스를 생성하며, 이로 인해 Invariant Core가 소실될 수 있음을 보였다.
3.  **경계 표류의 정량화**: 유도된 MDP 시퀀스 사이의 변화량을 측정하는 **Variation Budget ($V_E$)**을 도입하여, 경계의 불안정성과 구조적 재사용성 사이의 관계를 수식으로 연결하였다.
4.  **에이전트-세계 경계 프레임워크**: MARL의 비정상성(non-stationarity) 문제를 단순한 환경 변화가 아닌, 에이전트-세계 경계의 불안정성으로 인한 지속적 학습 문제로 재정의하였다.

## 📎 Related Works

논문은 다음과 같은 관련 연구들을 언급하며 차별점을 제시한다.

-   **표준 MDP 및 RL**: Sutton & Barto의 고전적 RL 프레임워크에서는 에이전트와 세계의 경계가 명확하며 역학이 정적이라고 가정한다.
-   **지속적 강화학습 (Continual RL)**: Khetarpal 등은 보상이나 역학이 시간에 따라 변하는 환경에서의 학습을 다루었다. 본 논문은 이러한 비정상성의 원인이 외부적인 작업 전환(exogenous task switches)뿐만 아니라, 에이전트-세계 경계의 설정에 따른 내생적 요인일 수 있음을 지적한다.
-   **다중 에이전트 강화학습 (MARL)**: 기존 MARL 연구들은 상대 에이전트의 정책 변화로 인한 비정상성을 다루어 왔으나, 본 논문은 이를 '에이전트-세계 경계'라는 모델링 선택의 관점에서 분석하여 CL 문제와 연결 지었다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. Trajectory Trie 및 Invariant Core
논문은 성공적인 궤적들의 집합 $S$를 기반으로 재사용 가능한 프로토타입을 정의한다.

-   **Trajectory Trie**: 상태-행동 쌍 $(s, a)$를 알파벳으로 하여 성공적인 궤적들을 트리 구조로 표현한다.
-   **Invariant Core**: 모든 성공적인 궤적 $\tau \in S$에 대해 공통적으로 나타나는 $\sqsubseteq$-최대 부분 시퀀스(maximal subsequences)들의 집합이다. 추상화 함수 $\phi: S \times A \to \Sigma$를 도입하여 더 의미 있는 단위(예: options)로 정의할 수 있으며, 수식은 다음과 같다.
$$\text{Core}_\phi(S) = \max_{\sqsubseteq} \{ u \in \Sigma^{\le H} : \forall \tau \in S, u \sqsubseteq \phi(\tau) \}$$
여기서 $u \sqsubseteq v$는 $u$가 $v$의 부분 시퀀스임을 의미한다.

### 2. 분산형 MARL에서의 유도된 MDP (Induced MDP)
두 에이전트가 참여하는 마르코프 게임에서, 초점 에이전트(focal agent)의 관점에서 상대 에이전트(agent 2)의 정책 $\pi_2^e$가 반영된 유도된 MDP $M^e$는 다음과 같이 정의된다.

-   **유도된 전이 확률 (Induced Transition Kernel)**:
$$ P^e(s'|s, a_1) = \sum_{a_2 \in A_2} P(s'|s, a_1, a_2) \pi_2^e(a_2|s) $$
-   **유도된 보상 함수 (Induced Reward Function)**:
$$ R^e(s, a_1) = \sum_{a_2 \in A_2} R_1(s, a_1, a_2) \pi_2^e(a_2|s) $$

상대 에이전트의 정책 $\pi_2^e$가 에피소드 $e$마다 업데이트되면, 초점 에이전트가 마주하는 MDP $M^e$는 계속해서 변화하며, 이에 따라 성공 궤적의 집합 $S^e$와 $\text{Core}_\phi(S^e)$ 또한 변화하게 된다.

### 3. Variation Budget ($V_E$)
에피소드 간의 경계 표류 정도를 정량화하기 위해 다음과 같은 Variation Budget을 정의한다.
$$ V_E = \mathbb{E} \sum_{e=2}^E \left( \sup_{s, a_1} \sum_{s'} |P^e(s'|s, a_1) - P^{e-1}(s'|s, a_1)| + \sup_{s, a_1} |R^e(s, a_1) - R^{e-1}(s, a_1)| \right) $$
이 값 $V_E = 0$이면 환경이 정적임을 의미하며, $V_E > 0$일 경우 상대 정책의 변화가 유도된 MDP의 변화를 일으켜 Invariant Core의 일부가 소실될 수 있음을 시사한다.

## 📊 Results

본 논문은 정량적인 실험 수치보다는 이론적인 증명과 분석 결과를 제시한다.

-   **정적 환경에서의 존재성 (Theorem 2.1)**: 고유한 흡수 목표(absorbing goal) $g$가 존재하거나 공통된 추상 심볼이 보장될 경우, $\text{Core}(S)$는 항상 존재한다.
-   **MARL에서의 Core 표류 (Proposition 2.1)**: 상대 에이전트의 정책 업데이트 $\pi_2^e \to \pi_2^{e+1}$에 의해, 에피소드 $e$에서 유효했던 프로토타입 $u \in \text{Core}_\phi(S^e)$가 에피소드 $e+1$에서는 더 이상 공통되지 않아 $\text{Core}_\phi(S^{e+1})$에서 사라질 수 있음을 보였다.
-   **정성적 예시**: 협력적 '열쇠-문(key-door)' 작업에서, 초기에는 '상대에게 열쇠를 전달 $\to$ 상대가 문으로 이동 $\to$ 상대가 문을 엶'이라는 구조가 Core에 포함될 수 있다. 그러나 상대 에이전트가 스스로 열쇠를 얻는 방법을 학습하면, 해당 프로토타입은 더 이상 모든 성공 궤적의 공통 요소가 아니게 되어 Core에서 사라진다.

## 🧠 Insights & Discussion

### 강점 및 이론적 의의
본 논문은 MARL의 비정상성 문제를 단순한 '학습의 어려움'이 아니라, **에이전트-세계 경계의 가변성**이라는 관점에서 분석하였다. 특히 'Invariant Core'라는 개념을 통해 무엇이 재사용 가능하고 무엇이 소실되는지를 이론적으로 정립한 점이 돋보인다. 이는 MARL을 CL의 특수한 사례로 해석함으로써, CL 분야의 기법들을 MARL의 안정성 향상에 적용할 수 있는 이론적 근거를 제공한다.

### 한계 및 논의사항
논문은 이론적 프레임워크를 제안하는 데 집중하고 있어, 실제 복잡한 환경에서 $V_E$를 실시간으로 추정하거나 이를 이용해 성능을 개선하는 알고리즘적 구현은 제시되지 않았다. 또한, 상대 에이전트의 정책 변화가 항상 Core를 소실시키는 것이 아니라, 오히려 더 단순하고 강력한 Core를 생성할 가능성에 대해서는 구체적으로 다루지 않았다.

### 비판적 해석
상대 에이전트를 세계의 일부로 보는 관점은 분석을 단순화하지만, 실제로는 상대 또한 학습하는 주체라는 점을 간과할 수 있다. 따라서 단순한 $V_E$ 측정보다는 상대의 정책 변화 방향을 예측하는 **Opponent Modeling**과의 결합이 필수적일 것으로 판단된다.

## 📌 TL;DR

본 논문은 성공적인 궤적들의 공통 구조인 **Invariant Core**를 정의하고, 분산형 MARL에서 상대 에이전트의 정책 업데이트가 에이전트-세계 경계를 불안정하게 만들어 이 Core를 소실시킨다는 것을 이론적으로 증명하였다. 이를 **Variation Budget ($V_E$)**으로 정량화함으로써, MARL 문제를 내생적 경계 표류에 의한 **지속적 학습(Continual Learning)** 문제로 재정의하였다. 이 연구는 향후 MARL에서 전이 학습의 실패 원인을 분석하고, 경계 표류에 강건한 옵션(options) 설계나 상대 모델링 연구에 중요한 이론적 토대를 제공한다.
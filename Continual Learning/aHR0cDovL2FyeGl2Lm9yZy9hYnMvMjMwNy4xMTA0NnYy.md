# 지속적 강화 학습에 대한 정의 (A Definition of Continual Reinforcement Learning)
David Abel, Andr ́e Barreto, Benjamin Van Roy, Doina Precup, Hado van Hasselt, Satinder Singh (Google DeepMind)

## 🧩 Problem to Solve
기존의 강화 학습(RL)은 에이전트가 장기 보상을 최대화하는 정책을 효율적으로 찾아낸 후 학습을 중단하는 "해결책을 찾는" 관점에 기반합니다. 그러나 이러한 관점은 실제 세계에서의 "끊임없는 적응"으로서의 학습을 충분히 포착하지 못합니다. 지속적 강화 학습(CRL)은 최고의 에이전트가 학습을 멈추지 않는 설정을 의미하지만, 이 중요한 개념에 대한 명확하고 간결한 정의가 부족합니다. 이 논문은 이러한 불분명성을 해소하고 CRL 문제에 대한 엄밀한 정의를 제공하는 것을 목표로 합니다.

## ✨ Key Contributions
*   **새로운 정의 언어 제시:** 에이전트가 "절대 학습을 멈추지 않는다"는 개념을 수학적으로 명확히 하기 위해 `generates` (생성) 및 `reaches` (도달)라는 두 가지 새로운 논리 연산자를 도입했습니다.
*   **에이전트의 행동 통찰력 제시:**
    *   모든 에이전트는 암묵적으로 다른 에이전트 집합(에이전트 기반)을 검색하는 것으로 이해될 수 있음을 보여줍니다 (Theorem 3.1).
    *   모든 에이전트의 검색은 영원히 계속되거나 결국 멈출 것임을 명확히 합니다 (Remark 3.2).
*   **지속적 학습 에이전트 정의:** 에이전트 기반($\Lambda_B$)에 대해 에이전트가 에이전트 기반을 생성($\Lambda_B \vdash_e \{\lambda\}$)하지만, 동시에 에이전트 기반에 절대 도달하지 않는($\lambda \not\mapsto_e \Lambda_B$) 경우를 "지속적 학습 에이전트"로 정의합니다 (Definition 4.1).
*   **지속적 강화 학습 (CRL) 문제 정의:** 주어진 RL 문제($e, v, \Lambda$)에서 모든 최적 에이전트($\Lambda^*$)가 특정 에이전트 기반($\Lambda_B$)에 대해 지속적 학습 에이전트일 경우, 해당 문제를 CRL 문제로 정의합니다 (Definition 4.2).
*   **동기 부여 예시 제시:** 기존의 다중 작업 RL(스위칭 MDP) 및 지속적 지도 학습이 제안된 CRL 정의의 특수한 경우임을 보여줍니다.
*   **CRL 및 연산자의 속성 분석:** CRL 문제의 필수 속성(Theorem 4.1)과 `generates` 및 `reaches` 연산자의 기본 속성(Theorem 4.2, Theorem 4.3)을 제시합니다. 특히, `generates` 및 `reaches` 결정 문제가 **비결정적(undecidable)**임을 보여줍니다.

## 📎 Related Works
이 논문의 CRL 개념은 수십 년간의 기계 학습 연구에서 중요한 부분이었던 "평생 학습(lifelong learning)" [63, 62, 53, 55, 51, 3, 4], "끝없는 학습(never-ending learning)" [39, 43]과 밀접한 관련이 있습니다. 또한 전이 학습(transfer learning) [61, 60], 메타 학습(meta-learning) [52, 17], 온라인 학습(online learning) 및 비정상성(non-stationarity) [5, 40, 13, 6, 31] 분야와도 연관됩니다.

특히, "지속적 작업(continuing tasks)"이라는 용어는 에이전트와 환경 간의 상호 작용이 에피소드(episodes)로 분할되지 않는 경우를 지칭하는 데 사용됩니다 [59]. 지속적 강화 학습은 링(Ring)의 논문에서 처음 제기되었으며 [46, 47, 48], 환경의 **일반성(generality)**에 중점을 둡니다. 최근 케타르팔 외(Khetarpal et al.) [25]는 CRL 문헌에 대한 포괄적인 조사를 제공하며, 기저 프로세스의 **비정상성(non-stationarity)**을 강조했습니다. 이 논문은 이러한 기존 연구의 일반성을 포용하면서도 "지속적 학습"의 의미에 대한 수학적 정의를 제공하여 더 큰 정밀성을 부여합니다.

## 🛠️ Methodology
이 논문은 CRL 문제를 정의하기 위해 에이전트와 환경의 상호 작용을 엄밀하게 모델링합니다.
1.  **기본 개념 정의:**
    *   **에이전트-환경 인터페이스:** 행동($\mathcal{A}$) 및 관찰($\mathcal{O}$) 집합으로 구성됩니다 (Definition 2.1).
    *   **히스토리($\mathcal{H}$):** 행동-관찰 쌍의 시퀀스로 정의됩니다 (Definition 2.2).
    *   **환경($e$):** 히스토리와 행동에 따라 관찰의 확률 분포를 생성하는 함수로 정의됩니다 ($e: \mathcal{H} \times \mathcal{A} \rightarrow \Delta(\mathcal{O})$) (Definition 2.3).
    *   **에이전트($\lambda$):** 히스토리에 따라 행동의 확률 분포를 선택하는 함수로 정의됩니다 ($\lambda: \mathcal{H} \rightarrow \Delta(\mathcal{A})$) (Definition 2.4). 이는 본질적으로 히스토리 기반 정책(history-based policy)입니다.
    *   **실현 가능 히스토리($\bar{\mathcal{H}}$):** 특정 에이전트-환경 쌍의 상호 작용으로 0이 아닌 확률로 발생할 수 있는 히스토리 집합입니다 (Definition 2.5).
    *   **보상 및 성능:** 보상 함수($r: \mathcal{A} \times \mathcal{O} \rightarrow \mathbb{R}$)와 성능 함수($v: \mathcal{H} \times \Lambda \times \mathcal{E} \rightarrow [v_{min}, v_{max}]$)를 정의합니다. 성능은 미래 보상의 통계량으로, 평균 보상 또는 할인된 보상으로 구체화될 수 있습니다 (Definition 2.7, 2.8).
    *   **RL 문제:** 주어진 환경($e$), 성능($v$), 사용 가능한 에이전트 집합($\Lambda$)에 대해 최적 에이전트($\Lambda^*$)를 찾는 문제로 정의됩니다 (Definition 2.9).
2.  **새로운 에이전트 연산자 정의:**
    *   **에이전트 기반($\Lambda_B$):** 에이전트의 비어 있지 않은 부분 집합입니다 (Definition 3.1).
    *   **학습 규칙($\sigma$):** 각 히스토리에 대해 기본 에이전트를 선택하는 함수입니다 ($\sigma: \mathcal{H} \rightarrow \Lambda_B$) (Definition 3.2).
    *   **`generates` ($\Lambda_B \vdash_e \Lambda$) 연산자 (Definition 3.4):** 에이전트 집합 $\Lambda$의 모든 에이전트가 환경 $e$의 실현 가능 히스토리에서 $\Lambda_B$의 요소들 사이를 전환하는 학습 규칙($\Sigma$)에 의해 생성될 수 있음을 나타냅니다. 즉, $\forall \lambda \in \Lambda \exists \sigma \in \Sigma \forall h \in \bar{\mathcal{H}}_{\lambda}(\lambda(h) = \sigma(h)(h))$.
    *   **$sometimes reaches$ ($\lambda \rightsquigarrow_e \Lambda_B$) 연산자 (Definition 3.5):** 특정 실현 가능 히스토리 $h$ 이후로 에이전트 $\lambda$의 행동이 $\Lambda_B$ 내의 특정 기본 에이전트 $\lambda_B$와 영원히 동일해지는 경우를 나타냅니다.
    *   **$never reaches$ ($\lambda \not\mapsto_e \Lambda_B$) 연산자 (Definition 3.6):** $sometimes reaches$의 부정으로, 에이전트가 어떤 기본 에이전트와도 같아지지 않는 경우를 나타냅니다.
3.  **CRL 문제의 공식화:** 위의 연산자들을 활용하여 최적 에이전트가 지속적 학습 에이전트(즉, 결코 검색을 멈추지 않는 에이전트)인 경우를 CRL 문제로 정의합니다 (Definition 4.2).

## 📊 Results
*   **스위칭 MDP에서의 Q-학습:** 환경이 동적으로 기저 MDP를 전환하고, 각 MDP는 고유한 최적 정책을 가집니다. 학습률($\alpha$)을 점차 줄여가는 수렴형 Q-학습(convergent Q-learning)은 환경이 전환될 때 이전에 최적이었던 정책이 더 이상 최적이 아니게 되므로, 학습을 멈추지 않고 일정한 $\alpha$를 유지하는 지속적 Q-학습(continual Q-learning)보다 성능이 낮습니다. 이는 최적 에이전트가 계속 적응해야 하는 CRL의 예시를 보여줍니다.
*   **지속적 지도 학습:** 분포 변화가 있는 이미지 분류와 같은 시나리오에서, 최적 에이전트는 변화하는 분포에 맞춰 지속적으로 분류기를 전환해야 합니다. 이는 에이전트가 특정 분류기에 수렴할 수 없는 CRL의 예시로 제시됩니다.
*   **CRL 문제의 필수 속성 (Theorem 4.1):**
    1.  CRL은 **기반 의존적(basis-dependent)**입니다. $\Lambda_B$의 변화에 따라 CRL 문제가 아닐 수 있습니다.
    2.  $\Lambda_B$의 어떤 요소도 최적일 수 없습니다 ($\Lambda_B \cap \Lambda^* = \emptyset$). 즉, 최적 에이전트가 되기 위해서는 기반 요소들 사이를 무기한 전환해야 합니다.
    3.  에이전트 집합 $\Lambda$는 **최소적(minimal)이지 않습니다.** 즉, CRL에서 에이전트 설계 공간에는 불필요한 부분이 존재합니다.
    4.  $\Lambda_B$는 계산 또는 메모리 예산과 같은 제약 조건으로 인해 제한될 수 있으며, 이는 **유한한(bounded) 에이전트**의 개념과 깊은 관련이 있음을 시사합니다.
*   **`generates` 연산자의 속성 (Theorem 4.2):**
    1.  **전이적(transitive)**입니다: $\Lambda_1 \vdash_e \Lambda_2$이고 $\Lambda_2 \vdash_e \Lambda_3$이면 $\Lambda_1 \vdash_e \Lambda_3$입니다.
    2.  **비교환적(not commutative)**입니다.
    3.  포함 관계에 따라 확장됩니다: $\Lambda_1^B \subseteq \Lambda_2^B$이고 $\Lambda_1^B \vdash_e \Lambda$이면 $\Lambda_2^B \vdash_e \Lambda$입니다.
    4.  모든 $\Lambda$와 $e$에 대해 $\Lambda \vdash_e \Lambda$입니다.
    5.  `generates` 결정 문제는 **비결정적(undecidable)**입니다 (정지 문제(Halting Problem)로의 환원).
*   **`reaches` 연산자의 속성 (Theorem 4.3):**
    1.  $sometimes reaches$ ($\mapsto_e$) 및 $never reaches$ ($\not\mapsto_e$)는 **비전이적(not transitive)**입니다.
    2.  $sometimes reaches$는 **비교환적(not commutative)**입니다.
    3.  $\lambda \in \Lambda$이면 $\lambda \not\mapsto_e \Lambda$입니다. (에이전트는 자기 자신을 포함하는 집합에 도달합니다.)
    4.  모든 에이전트는 모든 환경에서 $\Lambda$에 도달합니다.
    5.  `reaches` 결정 문제는 **비결정적(undecidable)**입니다 (정지 문제로의 환원).

## 🧠 Insights & Discussion
이 논문은 지속적 RL 문제에 대한 간단하지만 엄밀한 수학적 정의를 제시하여 AI 분야에서 그 중요성을 강조합니다. 모든 에이전트가 암묵적으로 에이전트 기반을 검색하며, 이 검색이 영원히 계속되거나 결국 멈춘다는 두 가지 통찰력을 바탕으로 `generates` 및 `reaches` 연산자를 통해 새로운 에이전트 이해 방식을 제공합니다.

**의미:**
*   **패러다임 전환:** 기존 RL이 문제의 고정된 해결책을 찾는 데 중점을 두는 반면, CRL은 경험에 따라 행동을 무기한 업데이트하는 에이전트 설계의 중요성을 강조합니다.
*   **기반의 중요성:** 에이전트 기반의 선택은 CRL 문제의 본질을 결정하며, 이는 실제 시스템 설계 시 계산 자원, 메모리 제약, 신경망 아키텍처 등과 같은 실용적인 고려 사항에 의해 결정됩니다.

**한계 및 향후 연구:**
*   **결정 문제의 비결정성:** `generates` 및 `reaches` 연산자와 관련된 결정 문제들이 대부분 비결정적이라는 결과는 일반적인 에이전트 집합과 환경에 대한 속성 결정의 어려움을 보여줍니다.
*   **특수 사례 탐색:** 이러한 결정 문제들이 결정 가능하고 효율적일 수 있는 흥미로운 특수 사례를 식별하는 것이 향후 연구 방향입니다.
*   **경험적 현상과의 연결:** 이 형식론을 최근 경험적 지속적 학습 연구의 핵심 현상들(예: 가소성 손실(plasticity loss), 인컨텍스트 학습(in-context learning), 치명적 망각(catastrophic forgetting))과 연결하는 것이 중요합니다.
*   **지속적 학습 규칙:** 지속적 학습 에이전트를 보장하는 "지속적 학습 규칙"의 특성을 정의하고, 이를 바탕으로 원칙적인 지속적 학습 알고리즘을 설계할 수 있습니다.

궁극적으로 이 논문의 정의, 분석 및 관점은 커뮤니티가 지속적 강화 학습을 새로운 시각으로 바라보고 새로운 연구 경로를 개척하는 데 도움이 될 것입니다.

## 📌 TL;DR
이 논문은 학습을 "끊임없는 적응"으로 보는 **지속적 강화 학습(CRL)**을 엄밀하게 정의합니다. 모든 에이전트가 암묵적으로 특정 에이전트 기반($\Lambda_B$)을 검색하며, 이 검색을 영원히 멈추지 않는 에이전트를 "지속적 학습 에이전트"로 정의합니다. CRL은 **최적의 에이전트가 모두 이러한 지속적 학습 에이전트인 RL 문제**입니다. 이 정의는 `generates`와 `reaches`라는 새로운 연산자를 통해 형식화되며, 기존의 다중 작업 RL 및 지속적 지도 학습이 이 정의의 특수 사례임을 보여줍니다. 주요 결과로는 CRL의 기반 의존성, 최적 에이전트가 $\Lambda_B$에 수렴하지 않는다는 점, 그리고 두 연산자 관련 결정 문제들이 **비결정적(undecidable)**이라는 점을 밝혀냈습니다. 이는 에이전트가 끊임없이 적응해야 하는 AI 시스템 설계에 대한 새로운 사고방식을 제시합니다.
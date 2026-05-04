# Large language models for artificial general intelligence (AGI): A survey of foundational principles and approaches

Alhassan Mumuni and Fuseini Mumuni (2025)

## 🧩 Problem to Solve

현대의 거대 언어 모델(Large Language Models, LLMs)과 멀티모달 모델들은 다양한 도메인에서 복잡한 문제를 해결하며 인상적인 성능을 보여주고 있다. 그러나 이러한 모델들의 인지 능력은 여전히 표면적이고 취약(brittle)하며, 학습 데이터에 존재하는 패턴을 모방하는 수준에 머물러 있다. 결과적으로 일반적인 LLM들은 진정한 의미의 범용성(generalist capabilities)을 갖추지 못하고 있다.

본 논문은 LLM이 인간 수준의 일반 지능(Human-level General Intelligence)에 도달하기 위해 해결해야 할 근본적인 문제로 **Embodiment(체화)**, **Symbol Grounding(기호 접지)**, **Causality(인과관계)**, 그리고 **Memory(메모리)**라는 네 가지 핵심 요소를 제시한다. 논문의 목표는 이러한 인지적 원칙들이 어떻게 AGI(Artificial General Intelligence) 구현의 토대가 되는지 분석하고, 이를 LLM에 통합하기 위한 최신 접근 방식들을 조사하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단순한 모델 규모의 확장(Scaling up)만으로는 AGI에 도달할 수 없음을 지적하고, 생물학적 지능의 특성을 모방한 네 가지 기초 원칙을 중심으로 한 **통합 인지 프레임워크(Unified Cognitive Framework)**를 제안한 것이다.

핵심 직관은 다음과 같다.

1. **Embodiment**: 지능은 신체와 환경의 상호작용을 통해 형성된다.
2. **Symbol Grounding**: 추상적인 기호(단어)를 실제 세계의 물리적 실체와 연결해야 한다.
3. **Causality**: 단순한 상관관계를 넘어 원인과 결과의 메커니즘을 이해해야 한다.
4. **Memory**: 습득한 지식과 경험을 체계적으로 저장하고 재구성하여 지속적 학습(Continual Learning)을 가능케 해야 한다.

## 📎 Related Works

논문은 AI의 발전 단계를 Narrow AI $\rightarrow$ Generative AI (VAEs, GANs) $\rightarrow$ Foundation Models (LLMs, VLMs, VLAs) 순으로 설명한다. 특히 최근의 멀티모달 LLM들이 AGI의 가능성을 열었으나, 여전히 다음과 같은 한계를 가진다고 분석한다.

- **기존 접근 방식의 한계**: 대부분의 LLM은 디지털 텍스트 데이터의 통계적 패턴에 의존하므로, 물리적 세계의 제약 조건이나 실제 인과 관계를 이해하지 못하는 '확률적 앵무새(Stochastic Parrots)'의 한계를 보인다.
- **AGI vs Strong AI**: 논문은 AGI와 Strong AI를 명확히 구분한다. AGI는 광범위한 인지 능력을 갖추고 복잡한 문제를 해결하는 '기능적 지능'에 집중하는 반면, Strong AI는 주관적 경험, 의식, 지향성(Intentionality)과 같은 '실제 인간의 인지적 속성'을 갖는 것을 목표로 한다.

## 🛠️ Methodology

본 논문은 AGI 구현을 위한 네 가지 핵심 원칙의 세부 메커니즘과 이를 LLM에 구현하는 방법론을 상세히 다룬다.

### 1. Embodiment (체화)

지능이 물리적 신체를 통해 환경과 상호작용하며 형성된다는 원칙이다.

- **주요 구성 요소**:
  - **Goal-awareness**: 외부 명령 없이도 내재된 상위 목표에 따라 자율적으로 행동하는 능력이다.
  - **Self-awareness**: 자신의 물리적 능력과 한계를 이해하고, 사회적 맥락에서 자신을 인식하는 능력이다.
  - **Situational-awareness**: 주변 환경의 상태와 타 지능체(Agent)의 의도를 파악하는 능력이다.
  - **Deliberate action**: 목적을 가지고 환경에 물리적 영향을 주는 행위이다.
- **구현 방법**: 실제 로봇에 LLM을 통합하거나, 가상 3D 환경(Game Engines, Physics Simulators)에서 에이전트를 학습시킨 후 실제 세계로 전이(Sim-to-Real)하는 방식을 사용한다.

### 2. Symbol Grounding (기호 접지)

추상적인 기호(심볼)를 실제 세계의 의미 있는 실체와 연결하는 과정이다.

- **구현 접근법**:
  - **Knowledge Graphs (KGs)**: 기호 간의 관계를 구조화하여 명시적 지식을 제공한다.
  - **Ontology-driven Prompting**: 온톨로지를 이용해 모델이 상황에 맞는 적절한 기호를 사용하도록 가이드한다.
  - **Vector Space Embeddings**: 고차원 벡터 공간에서 기호 간의 의미적 유사성을 학습한다.
  - **Active Exploration**: 강화 학습(RL) 등을 통해 환경과 직접 상호작용하며 기호의 의미를 체득한다.

### 3. Causality (인과관계)

단순한 상관관계(Correlation)가 아닌 인과적 메커니즘을 이해하는 능력이다.

- **인과성의 3단계 (Pearl's Hierarchy)**:
    1. **Association (Level 1)**: "X가 관찰되면 Y는 어떠한가?" (통계적 관계)
    2. **Intervention (Level 2)**: "X를 강제로 변화시키면 Y는 어떻게 변하는가?" (개입)
    3. **Counterfactual (Level 3)**: "만약 X가 다르게 일어났다면 Y는 어떻게 되었을까?" (가정/상상)
- **구현 방법**: SCM(Structural Causal Models)과 같은 인과 그래프를 LLM에 통합하거나, 물리 엔진 기반의 World Model을 통해 물리 법칙을 학습시킨다.

### 4. Memory (메모리)

지식을 보존, 공고화하고 재구성하는 시스템이다.

- **메모리 계층 구조**:
  - **Sensory Memory**: 외부 입력 신호를 일시적으로 버퍼링한다.
  - **Working Memory**: 현재 작업에 필요한 정보를 단기적으로 유지한다 (LLM의 Context Window가 이 역할을 수행한다).
  - **Long-term Memory**:
    - **Semantic Memory**: 일반적인 사실, 규칙, 개념 등에 대한 지식이다.
    - **Episodic Memory**: 개인적인 경험과 사건의 시퀀스를 저장한다.
    - **Procedural Memory**: 특정 작업을 수행하는 방법(How-to)에 대한 절차적 지식이다.
- **구현 방법**: 모델 파라미터 최적화, Attention 메커니즘, 외부 벡터 데이터베이스(Vector DB)를 이용한 RAG(Retrieval-Augmented Generation) 등이 사용된다.

## 📊 Results

본 논문은 특정 알고리즘의 성능을 측정하는 실험 논문이 아니라, 기존 연구들을 분석한 **서베이(Survey) 논문**이다. 따라서 정량적 결과 대신, 현재 기술 수준에 대한 분석 결과(State-of-the-art analysis)를 제시한다.

- **현재 LLM의 상태**: 멀티모달 LLM(예: PaLM-E, LLaVA)들이 상식 추론, 계획 수립 등에서 뛰어난 성능을 보이지만, 이는 대부분 학습 데이터의 패턴을 이용한 것이며 실제 물리적 법칙이나 인과 구조를 이해한 결과가 아님을 확인하였다.
- **Sim-to-Real Gap**: 가상 환경에서 학습된 에이전트가 실제 세계의 복잡성과 불확실성으로 인해 성능이 저하되는 문제가 여전히 존재한다.
- **메모리 한계**: LLM의 Context Window 크기가 증가하고 있으나, 'Lost in the Middle' 현상(입력값의 중간 부분을 망각하는 현상)과 같은 구조적 한계가 관찰된다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 AGI로 가는 경로를 단순한 '데이터 증량'이 아닌 '인지 구조의 완성' 관점에서 정의했다는 점에서 큰 의의가 있다. 특히 Embodiment $\rightarrow$ Grounding $\rightarrow$ Causality $\rightarrow$ Memory로 이어지는 논리적 흐름은 향후 AGI 아키텍처 설계의 가이드라인을 제공한다.

### 한계 및 비판적 해석

- **추상적 프레임워크**: 제안된 통합 프레임워크는 개념적 수준에 머물러 있으며, 이를 실제로 구현하기 위한 구체적인 수학적 모델이나 통합 아키텍처의 상세 설계도는 제시되지 않았다.
- **측정 가능성의 문제**: 논문에서도 언급되었듯이, '인간 수준의 지능'을 측정하는 객관적인 지표가 부재하므로, 제안된 원칙들이 실제로 얼마나 AGI에 근접하게 했는지 정량적으로 평가하기 어렵다.

### 종합 논의

결국 AGI의 핵심은 '디지털 세상의 통계'를 '물리 세상의 실체'로 변환하는 능력에 있다. Scaling Law가 일정 수준 이상의 효율 저하를 보일 때, 본 논문이 제시한 네 가지 원칙을 통합한 Neuro-symbolic 접근법이 돌파구가 될 가능성이 높다.

## 📌 TL;DR

본 논문은 LLM이 단순한 패턴 인식기를 넘어 **AGI(범용 인공 지능)**가 되기 위해서는 **체화(Embodiment), 기호 접지(Symbol Grounding), 인과관계 이해(Causality), 체계적 메모리(Memory)**라는 네 가지 인지적 기초가 필수적임을 논증한다. 저자들은 이 네 가지 요소가 서로 유기적으로 연결되어 지식을 습득하고 확장하는 **통합 인지 프레임워크**를 제안하며, 단순한 모델 확장보다는 생물학적 지능의 구조적 모방이 AGI 달성의 핵심이라고 주장한다.
